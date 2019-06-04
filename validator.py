import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from gene_cross import binarize
from sklearn import metrics
from sklearn.model_selection import train_test_split
import sys

if(len(sys.argv)<2):
    print("Not enough arguments. \n Needs 1 argument: \n\tfilename with values structured as \"results-UUID.py\"")
    sys.exit()

elif (len(sys.argv)>2):
    print("Too many arguments. \n Needs 1 argument: \n\tfilename with values structured as \"results-UUID.py\"")
    sys.exit()

else:
    print("Reading values from file: ", sys.argv[1])
    #get data
    exec(open(sys.argv[1]).read())

def printValues (f1, auc, acc, title):
    print("\n\n")
    print("--------------------------------")
    print(title)
    print("--------------------------------")
    print("F1 = ", f1)
    print("ACCURACY = ", acc)
    print("AUROC = ", auc)
    print("--------------------------------")


def fprintValues(f1, auc, acc, title, dataFileName,validateFileName, f):
    f.write("TEST/TRAIN:" + dataFileName + "\n")
    f.write("VALIDATION:" + validateFileName+ "\n")
    f.write("--------------------------------" + "\n")
    f.write(title + "\n")
    f.write("--------------------------------" + "\n")
    f.write("F1 = " + str(f1) + "\n")
    f.write("ACCURACY = " + str(acc) + "\n")
    f.write("AUROC = " + str(auc) + "\n")
    f.write("--------------------------------" + "\n")

path = r"../Data/JDT_R2_1.csv"
data = pd.read_csv(path, index_col=None, header=0)

X = data.loc[:, "SLOC_P":"MOD"]
y = data['bug_cnt']

binarize(y)
X_train, X_test, y_train, y_test = train_test_split(X, y)



mlp = MLPClassifier(hidden_layer_sizes=hiddenLayerSizes, max_iter=n_iter,
                    activation=activation_, alpha=alpha_,
                    verbose=True, solver="adam")

mlp.fit(X_train, y_train)
y_predicted = mlp.predict(X_test)

v_f1 = metrics.f1_score(y_test, y_predicted)
v_accuracy = metrics.accuracy_score(y_test, y_predicted)
v_auc = metrics.roc_auc_score(y_test, y_predicted)

d_f1 = v_f1 - F1
d_accuracy = v_accuracy - ACCURACY
d_auc = v_auc - AUC


printValues(v_f1, v_auc, v_accuracy, "VALIDATION SET")
printValues(F1, AUC, ACCURACY, "TEST SET")
printValues(d_f1, d_auc, d_accuracy, "DELTAS")

file = open(sys.argv[1][:-3]+"-"+path[8:-4]+".txt", 'w')
fprintValues(v_f1, v_auc, v_accuracy, "VALIDATION SET", sys.argv[1], path, file)
fprintValues(F1, AUC, ACCURACY, "TEST SET", sys.argv[1], path, file)
fprintValues(d_f1, d_auc, d_accuracy, "DELTAS", sys.argv[1], path, file)
file.close()