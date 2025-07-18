import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('heart_statlog_cleveland_hungary_final.csv')

df = df.drop_duplicates()

target_column = 'target'  # Change this if needed

# Separate features and target
X = df.drop(target_column, axis=1)
y = df[target_column]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Find optimal k value using cross-validation
print("\nFinding optimal k value...")
k_range = range(1, 31)
k_scores = []

for k in k_range:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_temp, X_train_scaled, y_train, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())

# Plot k vs accuracy
plt.figure(figsize=(10, 6))
plt.plot(k_range, k_scores, marker='o')
plt.xlabel('Value of k')
plt.ylabel('Cross-Validation Accuracy')
plt.title('KNN: Varying Number of Neighbors')
plt.grid(True)
plt.show()

# Find the best k
best_k = k_range[np.argmax(k_scores)]
best_score = max(k_scores)
print(f"Best k value: {best_k}")
print(f"Best cross-validation accuracy: {best_score:.4f}")

# Hyperparameter tuning with GridSearchCV
print("\nPerforming comprehensive hyperparameter tuning...")
param_grid = {
    'n_neighbors': range(3, 21),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

grid_search = GridSearchCV(
    KNeighborsClassifier(), 
    param_grid, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Train FINAL model with best parameters
final_knn = grid_search.best_estimator_
final_predictions = final_knn.predict(X_test_scaled)
final_accuracy = accuracy_score(y_test, final_predictions)

print(f"\n{'='*50}")
print(f"FINAL MODEL RESULTS")
print(f"{'='*50}")
print(f"Final Model Accuracy: {final_accuracy:.4f}")
print(f"Final Model Parameters: {grid_search.best_params_}")
print(f"\nFinal Model Classification Report:")
print(classification_report(y_test, final_predictions))

# Confusion Matrix for final model only
cm = confusion_matrix(y_test, final_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Disease', 'Disease'], 
            yticklabels=['No Disease', 'Disease'])
plt.title('Final Model - Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Cross-validation scores for final model
cv_scores = cross_val_score(final_knn, X_train_scaled, y_train, cv=5)
print(f"\nFinal Model Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Feature importance analysis
print(f"\n{'='*50}")
print(f"FEATURE ANALYSIS")
print(f"{'='*50}")
feature_corr = X.corrwith(y).abs().sort_values(ascending=False)
print("Top 5 most correlated features:")
print(feature_corr.head())

# Save the trained model and scaler
import joblib
joblib.dump(final_knn, 'heart_disease_knn_model.pkl')
joblib.dump(scaler, 'heart_disease_scaler.pkl')
print(f"\nModel and scaler saved successfully!")

# Example prediction function
def predict_heart_disease(input_features):
    """
    Make a prediction for heart disease based on input features.
    input_features: list or array of feature values in the same order as training data
    """
    # Scale the input
    input_scaled = scaler.transform([input_features])
    
    # Make prediction
    prediction = final_knn.predict(input_scaled)[0]
    probability = final_knn.predict_proba(input_scaled)[0]
    
    return prediction, probability

print(f"\nModel training completed successfully!")
print(f"Use predict_heart_disease(features) to make predictions on new data.")