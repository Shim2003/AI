import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Load and clean data
data = pd.read_csv('cleaned_heart_disease_data.csv')
cleanedData = data.drop_duplicates().dropna()

# 2. Separate features and target
X = cleanedData.drop('target', axis=1)
y = cleanedData['target']

# 3. List of categorical columns to encode
categorical_cols = ['chest pain type', 'resting ecg', 'ST slope']

# 4. Define preprocessing steps
preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

# 5. Create full pipeline: preprocess → scale → logistic regression
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('scale', StandardScaler(with_mean=False)),  # Use with_mean=False for sparse matrix
    ('logreg', LogisticRegression(max_iter=1000, random_state=42))
])

# 6. Cross-validation (10-fold)
cv_scores = cross_val_score(pipeline, X, y, cv=10, scoring='accuracy')
print("Cross-Validation Accuracy Scores:", cv_scores)
print("Average CV Accuracy:", cv_scores.mean())
print("Standard Deviation:", cv_scores.std())

# 7. Train-test split for final evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)

# 8. Train model
pipeline.fit(X_train, y_train)

# 9. Predict and evaluate
y_pred = pipeline.predict(X_test)

print("\n--- Final Evaluation on Test Set ---")
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 10. Save the model
joblib.dump(pipeline, 'heart_disease_logreg_crossValidation_model.pkl')

# clean and remove the "0" outliers which is only 1 and it's irrelevant
# print(cleanedData['ST slope'].value_counts())
# print(cleanedData[['ST slope', 'target']].groupby('ST slope').mean())

# print(cleanedData['sex'].value_counts())
# print(cleanedData[['sex', 'target']].groupby('sex').mean())

# print(cleanedData['chest pain type'].value_counts())
# print(cleanedData[['chest pain type', 'target']].groupby('chest pain type').mean()) 
# typical angina 一般心绞痛， atypical angina 非一般心绞痛， non-anginal pain 非心绞痛疼痛， asymptomatic pain 无症状疼痛
# chest pain type          
# 1                0.434783
# 2                0.138728
# 3                0.351485
# 4                0.790323

# print(cleanedData['oldpeak'].describe())
# print(cleanedData.groupby('target')['oldpeak'].mean())  # 0: 0.421359, 1: 1.314651
# print(cleanedData.groupby('target')['oldpeak'].median()) #0: 0.0, 1: 1.2

# sns.boxplot(x='target', y='oldpeak', data=cleanedData)
# plt.xlabel("Heart Disease (1 = Yes, 0 = No)")
# plt.ylabel("Oldpeak")
# plt.title("Oldpeak Distribution by Heart Disease")
# plt.show()     #超过1会比较有风险，一点点

# print(cleanedData.columns)