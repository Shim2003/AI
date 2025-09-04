import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Load and clean data
cleanedData = pd.read_csv('cleaned_heart_disease_data.csv')

# 2. Separate features and target
X = cleanedData.drop('target', axis=1)
y = cleanedData['target']

# 3. List of categorical and skewed columns
categorical_cols = ['chest pain type', 'resting ecg', 'ST slope']
skewed_cols = ['cholesterol', 'oldpeak']  # apply log transform

# 4. Define preprocessing steps
preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('log', FunctionTransformer(np.log1p, validate=False), skewed_cols)
], remainder='passthrough')

# 5. Build pipeline: preprocess → scale → logistic regression
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('scale', StandardScaler(with_mean=False)),  # for sparse matrices
    ('logreg', LogisticRegression(max_iter=1000, random_state=42))
])

# 6. Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# 7. Hyperparameter tuning with GridSearchCV
param_grid = {
    'logreg__C': [0.01, 0.1, 1, 10, 100],
    'logreg__penalty': ['l1', 'l2'],  # L1 = Lasso, L2 = Ridge
    'logreg__solver': ['liblinear']   # supports both L1 and L2
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=10,
    scoring='accuracy',
    n_jobs=-1
)

grid.fit(X_train, y_train)

# 8. Best model from GridSearch
best_model = grid.best_estimator_
print("Best Params:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)

# 9. Evaluate on hold-out test set
y_pred = best_model.predict(X_test)

print("\n--- Final Evaluation on Test Set ---")
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 10. Save the best model
joblib.dump(best_model, 'heart_disease_logreg_model.pkl')
print("\nModel saved as heart_disease_logreg_model.pkl")


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


# Classification Report:
#                precision    recall  f1-score   support

#            0       0.88      0.85      0.86        82
#            1       0.88      0.90      0.89       102

#     accuracy                           0.88       184
#    macro avg       0.88      0.88      0.88       184
# weighted avg       0.88      0.88      0.88       184

# Barplot: Proportion of heart disease by sex
# sns.barplot(x="sex", y="target", data=cleanedData, ci=None)

# plt.xlabel("Sex (0 = Female, 1 = Male)")
# plt.ylabel("Proportion with Heart Disease")
# plt.title("Heart Disease Prevalence by Sex")
# plt.ylim(0, 1)  # since it's proportion
# plt.show()