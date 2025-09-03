import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# read CSV file
df = pd.read_csv('cleaned_heart_disease_data.csv')

# Separate features and labels
X = df.drop('target', axis=1)
y = df['target']

# Split training/testing dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM model with RBF kernel and probability output
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Make predictions and evaluate
y_pred = svm_model.predict(X_test_scaled)
print(f"\n{'='*50}")
print("🎯 SVM Model Evaluation")
print(f"{'='*50}")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save model & scaler
joblib.dump(svm_model, 'heart_disease_svm_model.pkl')
joblib.dump(scaler, 'heart_disease_scaler.pkl')
print(" Model and scaler saved successfully！")

# Prediction function
def predict_svm(input_features):
    model = joblib.load('heart_disease_svm_model.pkl')
    scaler = joblib.load('heart_disease_scaler.pkl')
    input_scaled = scaler.transform([input_features])
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    return prediction, probability
