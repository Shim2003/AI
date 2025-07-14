import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('heart_statlog_cleveland_hungary_final.csv')

print(data.head())