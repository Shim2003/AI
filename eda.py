import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('cleaned_heart_disease_data.csv')  # Replace with your actual file path

# Numerical features distribution
num_features = df.select_dtypes(include=['float64','int64']).columns.drop('target')

df[num_features].hist(bins=20, figsize=(15, 10))
plt.suptitle("Numerical Feature Distributions")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
