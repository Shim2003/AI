import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Safely read CSV
csv_path = os.path.join(os.path.dirname(__file__), 'heart_statlog_cleveland_hungary_final.csv')
df = pd.read_csv(csv_path).drop_duplicates()

# Features and target
X = df.drop('target', axis=1)
y = df['target']

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='accuracy',
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)
rf_model = grid_search.best_estimator_

# Make predictions
y_pred = rf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

# Evaluation results
print("\n" + "="*50)
print("🌲 Random Forest Model Evaluation (Using Best Parameters)）")
print("="*50)
print("🎯 Best Parameters:", grid_search.best_params_)
print(f"🎯 Accuracy: {accuracy:.4f}")
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Save model & scaler
joblib.dump(rf_model, 'heart_disease_rf_model.pkl')
joblib.dump(scaler, 'heart_disease_rf_scaler.pkl')
print(" Model and scaler saved successfully!")

# Prediction function
def predict_rf(input_features):
    model = joblib.load('heart_disease_rf_model.pkl')
    scaler = joblib.load('heart_disease_rf_scaler.pkl')
    input_scaled = scaler.transform([input_features])
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    return prediction, probability
