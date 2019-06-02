import pandas as pd
import glob
import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from gene_cross import *
from sklearn import metrics
import math
import matplotlib.pyplot as plt
import uuid

POPULATION_SIZE = 100
ITERATIONS = 200
PATH = r"Data/JDT_R2_0.csv"

NOTE = "Fitness used: $\sqrt{\\frac{F_1 ^ 2 + S_{AUROC} ^ 2 + Accuracy ^ 2}{3}}$"
print(NOTE+"\n")

def rouletteWheelSelection(population):
    max = sum([agent.fitness for agent in population])
    probs = [agent.fitness/max for agent in population]
    return population[random.choice(len(population), p=probs)]
 

data = pd.read_csv(PATH, index_col = None, header = 0)
"""
#use for reading ALL the data
PATH = r'Data/'
dataSets = glob.glob(os.PATH.join(PATH, "*.csv"))

data = pd.concat((pd.read_csv(d, index_col=None, header=0)
                for d in dataSets), axis=0, ignore_index=True)
"""
X = data.loc[:, "SLOC_P":"MOD"]
y = data['bug_cnt']

binarize(y)

X_train, X_test, y_train, y_test = train_test_split(X, y)
population = []

#generate population START
for i in range(POPULATION_SIZE):
    g1 = Gene([0], '.', 0.0, 0)
    g1.setRandom()
    population.append(g1)

fit = []
fit_accuracy = []
fit_f1 = []
fit_auc = []

for n in range(ITERATIONS):
    print("------------------\nITERATION =", n+1,"/",ITERATIONS, "\n------------------\n")
    l=0
    for i in range(POPULATION_SIZE):
        l+=1
        print((int(((l-1)/POPULATION_SIZE)*100)),"%", end="\r")
        #if fitnes has already been calculated skip that agent
        if population[i].f1!=0.0  and population[i].auc!=0.0 and population[i].accuracy != 0.0 and n>0:
            continue

        mlp = MLPClassifier(hidden_layer_sizes=population[i].hiddenLayerSizes, max_iter=population[i].n_iter,
                            activation=population[i].activationFunction, alpha=population[i].alpha,
                            verbose=False, solver="adam")

        mlp.fit(X_train,y_train)
        y_predicted = mlp.predict(X_test)

        population[i].f1 = metrics.f1_score(y_test, y_predicted)
        population[i].auc = metrics.accuracy_score(y_test, y_predicted)
        population[i].accuracy = metrics.roc_auc_score(y_test, y_predicted)
        
        #######################################################
        #CHANGE METRIC USED FOR FITNESS HERE
        #population[i].fitness = population[i].auc
        #population[i].fitness = population[i].f1
        #population[i].fitness = population[i].accuracy
        population[i].fitness = np.sqrt(np.square(population[i].auc)+np.square(population[i].f1)+np.square(population[i].accuracy))/np.sqrt(3)
        ########################################################

        print((int((l/POPULATION_SIZE)*100)), "% :", "|",
                "F1 = ", '{0:5f}'.format(population[i].f1), "|",
                "AUC = ", '{0:5f}'.format(population[i].auc), "| "
                "ACCURACY = ", '{0:5f}'.format(population[i].accuracy), end="\r")
    
    #SINGLE OBJECTIVE
    #population.sort(key=lambda x: x.fitness, reverse=True)
    #MULTI OBJECTIVE
    population.sort(key=lambda x: x.fitness, reverse=True)
    fit.append(population[0].fitness)
    fit_accuracy.append(population[0].accuracy)
    fit_f1.append(population[0].f1)
    fit_auc.append(population[0].auc)

    #delete worse half of survivors
    del population[-math.floor(len(population)/2):]

    #refill the population
    for i in range(POPULATION_SIZE-len(population)):
        #pick 2 genes at random
        gene1 = rouletteWheelSelection(population)
        gene2 = rouletteWheelSelection(population)
        #if the selected genes are the same select a new gene
        while True:
            if gene1 == gene2:
                gene2 = rouletteWheelSelection(population)
            else:
                break
        
        newGene = crossGenes(gene1,gene2)
        population.append(newGene)

    print("Done. ", "|",
          "F1 = ", '{0:5f}'.format(population[0].f1), "|",
          "AUC = ", '{0:5f}'.format(population[0].auc), "|"
          "ACCURACY = ", '{0:5f}'.format(population[0].accuracy), end="\r")
    print("\n\n")
    


filename = str(uuid.uuid4())

file = open("results-"+filename+"-"+PATH[5:-4]+".py", 'w')

#print values
print("\n----------------\nBest Solution Found\n-----------------\n")
population[0].printAll()

delta = abs(fit[len(fit)-1]-fit[0])
print("DELTA:", delta)
#write them to file
population[0].fprintAll(file)

file.write("DELTA = " + str(delta))

plt.figure()
plt.title("Score changes on dataset " + PATH[5:-4] + "\nIterations: "  \
            + str(ITERATIONS) + " Population: " + str(POPULATION_SIZE) \
            + " Mutation: " + '{0:1f}'.format(1/MUTATION_CHANCE*100) + "%" \
            + "\n" + NOTE)
plt.plot(np.arange(len(fit_f1)), fit_f1, label='F1')
plt.plot(np.arange(len(fit_accuracy)), fit_accuracy, label='Accuracy')
plt.plot(np.arange(len(fit_auc)), fit_auc, label='AUROC')

plt.plot(np.arange(len(fit)), fit,
         label="$\sqrt{\\frac{F_1 ^ 2 + S_{AUROC} ^ 2 + Accuracy ^ 2}{3}}$")

plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend(loc=0)

# Put a legend to the right of the current axis
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig("results-"+filename+"-"+PATH[5:-4]+".png")
plt.show()

file.close()

