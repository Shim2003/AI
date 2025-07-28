import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression           for testing split data accuracy
# from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler


data = pd.read_csv('heart_statlog_cleveland_hungary_final.csv')

# print(data.head())

# # Display basic info
# print(data.info())

# Remove duplicate rows and missing if any
cleanedData = data.drop_duplicates().dropna()

# print("Missing:", data.isnull().sum())  #no missing values

# Check for duplicates and nulls again
print("Data before cleaning:", len(data))  #1190
print("Duplicates removed:", data.duplicated().sum())   #272
print("Data left after cleaning:", len(cleanedData))  #918

# Separate features and target
X = cleanedData.drop('target', axis=1)
y = cleanedData['target']

# from nominal to numeric
# List of categorical columns to one-hot encode
categorical_cols = ['chest pain type', 'resting ecg', 'ST slope']

# Apply OneHotEncoding
ct = ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(), categorical_cols)
], remainder='passthrough')

X_encoded = ct.fit_transform(X)

# Split the data  (90% training, 10% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y)  # stratify to preserve class ratio

# Build ANN using MLPClassifier
mlp = MLPClassifier(
    hidden_layer_sizes=(16, 8),  # Two hidden layers
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)

# Train the model
mlp.fit(X_train, y_train)

# Make predictions
y_pred = mlp.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# Save the trained model
joblib.dump(mlp, 'heart_disease_ann_model.pkl')

# Load the saved model
loaded_model = joblib.load('heart_disease_ann_model.pkl')

# Use it to predict
predictions = loaded_model.predict(X_test)
print("Loaded model accuracy:", accuracy_score(y_test, predictions))

# # Print shape
# print("Training set size:", X_train.shape)
# print("Testing set size:", X_test.shape)

# sns.countplot(x='target', data=cleanedData)           graph/plotting
# plt.title('Heart Disease Distribution')
# plt.show()

# # Try multiple test sizes
# test_sizes = [0.1, 0.2, 0.3]  # 90/10, 80/20, 70/30

# for test_size in test_sizes:
#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, random_state=42, stratify=y)

#     # Train model
#     model = LogisticRegression(max_iter=1000)
#     model.fit(X_train, y_train)

#     # Predict and evaluate
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)

#     print(f"Test size: {int(test_size*100)}%, Accuracy: {accuracy:.4f}")

# # Test size: 10%, Accuracy: 0.8804
# # Test size: 20%, Accuracy: 0.8804
# # Test size: 30%, Accuracy: 0.8696