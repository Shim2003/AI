import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression           
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Load and clean data
data = pd.read_csv('heart_statlog_cleveland_hungary_final.csv')
cleanedData = data.drop_duplicates().dropna()

# 2. Separate features and target
X = cleanedData.drop('target', axis=1)
y = cleanedData['target']

# 3. List of categorical columns to encode
categorical_cols = ['chest pain type', 'resting ecg', 'ST slope']

# 4. Define preprocessing steps
preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(), categorical_cols)
], remainder='passthrough')

# 5. Create full pipeline: preprocess → scale → logistic regression
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('scale', StandardScaler(with_mean=False)),  # with_mean=False for sparse matrix
    ('logreg', LogisticRegression(max_iter=1000, random_state=42))
])

# 6. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)

# 7. Train model
pipeline.fit(X_train, y_train)

# 8. Predict and evaluate
y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 9. Save the model
joblib.dump(pipeline, 'heart_disease_logreg_model.pkl')

# Optional: Load it back to use later
# loaded_model = joblib.load('heart_disease_logreg_model.pkl')
# predictions = loaded_model.predict(X_test)
