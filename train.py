import pandas as pd
import glob
import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from gene_cross import *
from sklearn import metrics
import math



POPULATION_SIZE = 20
ITERATIONS = 20
path = r"Data/JDT_R2_0.csv"
data = pd.read_csv(path, index_col = None, header = 0)
"""
#use for reading ALL the data
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
fit_rmse = []
fit_f1 = []
fit_r2 = []

for n in range(ITERATIONS):
    print("------------------\nITERATION =", n, "\n------------------\n")
    l=0
    for i in range(POPULATION_SIZE):
        l+=1
        print((l/POPULATION_SIZE)*100,"%")
        #if fitnes has already been calculated skip that agent
        #if population[i].fitness[0]!=0 and n>0:
        #    continue


        mlp = MLPClassifier(hidden_layer_sizes=population[i].hiddenLayerSizes, max_iter=population[i].n_iter,
                            activation=population[i].activationFunction, alpha=population[i].alpha,
                            verbose=False, solver="adam")


        mlp.fit(X_train,y_train)
        y_predicted = mlp.predict(X_test)
        #population[i].fitness = 
        
        #f1Score = metrics.f1_score(y_test, y_predicted, average='weighted')
        population[i].fitness[0] = metrics.mean_squared_error(y_test, y_predicted)
        population[i].fitness[1] = metrics.f1_score(y_test, y_predicted, average='weighted')
        population[i].fitness[2] = metrics.r2_score(y_test, y_predicted)

    population.sort(key=lambda x: x.fitness[0], reverse=False)
    fit.append(population[0].fitness[0])
    fit_rmse.append(population[0].fitness[0])
    fit_f1.append(population[0].fitness[1])
    fit_r2.append(population[0].fitness[2])
    #delete worse half of survivors
    del population[-math.floor(len(population)/2):]
    #refill the population
    for i in range(POPULATION_SIZE-len(population)):
        #pick 2 genes at random
        #generate new genes by crossing and mutating genes
        gene1=population[random.randint(len(population))]
        gene2 = population[random.randint(len(population))]

        newGene = crossGenes(gene1,gene2)

        population.append(newGene)
        print("LEN=",len(population))

    
import matplotlib.pyplot as plt

#print values
print("\n----------------\nBest Solution Found\n-----------------\n")
population[0].printAll()
print("Fitness:", population[0].fitness)
delta = abs(fit[len(fit)-1]-fit[0])
print("DELTA:", delta)

plt.plot(np.arange(len(fit_rmse)),fit_rmse, label='RMSE')
plt.plot(np.arange(len(fit_f1)), fit_f1, label='F1')
plt.plot(np.arange(len(fit_r2)), fit_r2, label='R2')
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend(loc='upper left')

plt.show()
