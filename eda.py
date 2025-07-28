import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('heart_statlog_cleveland_hungary_final.csv')  # Replace with your actual file path
df.head()
df.info()
df.describe()

# df['target'].value_counts()
# sns.countplot(x='target', data=df)
# plt.title('Target Class Distribution')
# plt.show()

# df.hist(bins=15, figsize=(15, 10))
# plt.tight_layout()
# plt.show()

categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
for col in categorical_cols:
    sns.countplot(x=col, data=df)
    plt.title(f'Distribution of {col}')
    plt.show()

