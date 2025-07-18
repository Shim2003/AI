import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('heart_statlog_cleveland_hungary_final.csv')

# print(data.head())

# # Display basic info
# print(data.info())

# Remove duplicate rows and missing if any
cleanedData = data.drop_duplicates().dropna()

print("Missing:", data.isnull().sum())  #no missing values

# Check for duplicates and nulls again
print("Data before cleaning:", len(data))  #1190
print("Duplicates removed:", data.duplicated().sum())   #272
print("Data left after cleaning:", len(cleanedData))  #918

# Separate features and target
X = cleanedData.drop('target', axis=1)
y = cleanedData['target']

# Split the data  (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)  # stratify to preserve class ratio

# Print shape
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

sns.countplot(x='target', data=cleanedData)
plt.title('Heart Disease Distribution')
plt.show()
