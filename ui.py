import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("heart_disease_logreg_crossValidation_model.pkl")

st.title("💓 Heart Disease Prediction App")

st.write("Fill in the patient details to predict the likelihood of heart disease.")

# User inputs for all features
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
chest_pain = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4])
resting_bp = st.number_input("Resting Blood Pressure (mmHg)", min_value=50, max_value=250, value=120)
cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
resting_ecg = st.selectbox("Resting ECG", options=[0, 1, 2])
max_hr = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=150)
exercise_angina = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
# Let user pick from meaningful labels
st_slope_label = st.selectbox(
    "ST Slope",
    options=["Upsloping (1)", "Flat (2)", "Downsloping (3)"]
)

# Convert back to numeric value for the model
st_slope = int(st_slope_label.split("(")[1][0])

# Predict button
if st.button("Predict"):
    # Define columns in the same order as training dataset
    columns = ['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol',
               'fasting blood sugar', 'resting ecg', 'max heart rate',
               'exercise angina', 'oldpeak', 'ST slope']
    
    # Wrap input data in a DataFrame
    input_df = pd.DataFrame([[
        age, sex, chest_pain, resting_bp, cholesterol,
        fasting_bs, resting_ecg, max_hr,
        exercise_angina, oldpeak, st_slope
    ]], columns=columns)
    
    # Now predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    # Output
    if prediction == 1:
        st.error(f"⚠️ High likelihood of Heart Disease (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low likelihood of Heart Disease (Probability: {probability:.2f})")
