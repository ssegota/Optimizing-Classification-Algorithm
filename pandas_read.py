import pandas as pd
import glob
import os
#require pandas==0.24.2
#load all CSV
path = r'../Data/'
dataSets = glob.glob(os.path.join(path, "*.csv"))

df = pd.concat((pd.read_csv(d, index_col=None, header=0) for d in dataSets), axis=0, ignore_index=True)

#print columns
print(df.columns)

#get a row
#row = 19351
#print(df.iloc[row,:])

#get a column

#print(df.loc[:,"bug_cnt"])
df_bugs = df.loc[:, "bug_cnt"]
df_bugs.to_csv('bugs.csv', header=True)

#get everything but bug count

df_metrics = df.loc[:, df.columns != "bug_cnt"]
#
#print(df.loc[:, df.columns !="bug_cnt"])
df_metrics.to_csv('metrics.csv')
#print(df_metrics.columns)
print("Number of metrics:", len(df_metrics.columns))

#convert metrics to numpy array
metrics = df.loc[:,"LOC":"MOD"]
metrics = metrics.to_numpy()
print(metrics)

#https://stackoverflow.com/questions/11023411/how-to-import-csv-data-file-into-scikit-learn
