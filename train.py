import pandas as pd
import glob
import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from gene_cross import *
from sklearn import metrics
import math

POPULATION_SIZE = 100

path = r"Data/JDT_R2_0.csv"
data = pd.read_csv(path, index_col = None, header = 0)
"""
path = r'Data/'
dataSets = glob.glob(os.path.join(path, "*.csv"))

data = pd.concat((pd.read_csv(d, index_col=None, header=0)
                for d in dataSets), axis=0, ignore_index=True)
"""
X = data.loc[:, "SLOC_P":"MOD"]
y = data['bug_cnt']

X_train, X_test, y_train, y_test = train_test_split(X, y)


population = []
#generate population START
for i in range(POPULATION_SIZE):
    g1 = Gene([0], '.', '.', 0.0, 0)
    g1.setRandom()
    population.append(g1)

fit = []

for n in range(100):
    print("------------------\nITERATION =", n, "\n------------------\n")
    l=0
    for i in range(POPULATION_SIZE):
        l+=1
        print((l/POPULATION_SIZE)*100,"%")
        print(i, len(population))
        if population[i].fitness!=0 and n>0:
            continue


        mlp = MLPClassifier(hidden_layer_sizes=population[i].hiddenLayerSizes, max_iter=population[i].n_iter,
                            activation=population[i].activationFunction, alpha=population[i].alpha,
                            verbose=False, solver="adam")


        mlp.fit(X_train,y_train)
        y_predicted = mlp.predict(X_test)
        #Eliminate all 0 predictions?
        """
        if sum(y_predicted == 0):
            population[i].fitness = -1
            continue
        """
        population[i].fitness = metrics.f1_score(y_test, y_predicted, average='weighted')
        #print("Fitness of gene", i+1,"=",population[i].fitness, "\nsolution SUM=", sum(y_predicted))


        #print(metrics.classification_report(y_test, y_predicted))


    population.sort(key=lambda x: x.fitness, reverse=True)
    #delete all 0 predictions
    """
    k_=[]
    for k in range(len(population)):
        if population[k].fitness==-1:
            k_.append(k)

    for k in k_:
        population[k] = Gene([0], '.', '.', 0.0, 0)
        population[k].setRandom()
    """
    """
    print("Units in population=",len(population))
    print(".................\nPopulation sorted by Fitness:\n......................")
    for k in population:
        print("--------------------")
        k.printAll()
        print("Fitness:", k.fitness)
    
    """
    fit.append(population[0].fitness)
    #delete worse half of survivors
    del population[-math.floor(len(population)/2):]
    #refill the population
    for i in range(POPULATION_SIZE-len(population)):
        #pick 2 genes at random
        #generate new genes by crossing and mutating genes
        gene1=population[random.randint(len(population))]
        gene2 = population[random.randint(len(population))]
        #newGene = Gene([0], '.', '.', 0.0, 0)
        #newGene.setRandom()
        newGene = crossGenes(gene1,gene2)

        population.append(newGene)
        print("LEN=",len(population))

    
import matplotlib.pyplot as plt

#print(np.arange(len(fit)))
#print(fit)

#print values
print("\n----------------\nBest Solution Found\n-----------------\n")
population[0].printAll()
print("Fitness:", population[0].fitness)
delta = abs(fit[len(fit)-1]-fit[0])
print("DELTA:", delta)

plt.plot(np.arange(len(fit)),fit, label='F1 Score')
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend(loc='upper left')

plt.show()
