import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#DATA LOADING
df = pd.read_csv('heart_statlog_cleveland_hungary_final.csv')

print(df.head())

row_count = len(df)
print(f"Total data rows (excluding header): {row_count}")


#DATA CLEANING

#checking for null values
print(df.isnull().sum())

#checking for duplicates data
duplicates = df[df.duplicated(keep=False)]
print(duplicates)

df = df.drop_duplicates()

print(df.duplicated().sum())

row_count = len(df)
print(f"Total data rows (excluding header): {row_count}")

#EDA
import matplotlib.pyplot as plt
import seaborn as sns

# Correlation heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Boxplot to visualize scale differences
numeric_columns = ['age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']

plt.figure(figsize=(10, 6))
df[numeric_columns].boxplot()
plt.title('Boxplot of Numeric Features')
plt.ylabel('Value Range')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Also print basic statistics to support the plot
print("Feature Ranges (Summary):")
print(df[numeric_columns].describe())

#ui