import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("final_knn_model.pkl")

model = load_model()

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = {}

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Heart Disease Prediction", "Lifestyle Recommendations"])

# Rules-based recommendation engine
def get_lifestyle_recommendations(user_data, prediction_prob):
    recommendations = []
    
    # Age-based recommendations
    if user_data['age'] > 50:
        recommendations.append({
            'category': 'Age Management',
            'icon': '👴',
            'advice': 'Regular health check-ups are crucial after 50. Schedule annual cardiovascular screenings.',
            'priority': 'High'
        })
    
    # Blood pressure recommendations
    if user_data['resting_bp'] > 140:
        recommendations.append({
            'category': 'Blood Pressure',
            'icon': '🩺',
            'advice': 'Your blood pressure is elevated. Reduce sodium intake to less than 2,300mg daily and consider the DASH diet.',
            'priority': 'High'
        })
    elif user_data['resting_bp'] > 120:
        recommendations.append({
            'category': 'Blood Pressure',
            'icon': '🩺',
            'advice': 'Monitor your blood pressure regularly. Limit salt and increase potassium-rich foods like bananas and leafy greens.',
            'priority': 'Medium'
        })
    
    # Cholesterol recommendations
    if user_data['cholesterol'] > 240:
        recommendations.append({
            'category': 'Cholesterol',
            'icon': '🥗',
            'advice': 'High cholesterol detected. Adopt a low-saturated fat diet, increase fiber intake, and consider omega-3 supplements.',
            'priority': 'High'
        })
    elif user_data['cholesterol'] > 200:
        recommendations.append({
            'category': 'Cholesterol',
            'icon': '🥗',
            'advice': 'Borderline high cholesterol. Include oats, beans, and nuts in your diet. Limit red meat and processed foods.',
            'priority': 'Medium'
        })
    
    # Fasting blood sugar recommendations
    if user_data['fasting_bs'] == 1:
        recommendations.append({
            'category': 'Blood Sugar',
            'icon': '🍎',
            'advice': 'Elevated blood sugar detected. Monitor carbohydrate intake, choose complex carbs, and maintain regular meal times.',
            'priority': 'High'
        })
    
    # Heart rate recommendations
    if user_data['max_hr'] < 100:
        recommendations.append({
            'category': 'Physical Fitness',
            'icon': '🏃',
            'advice': 'Low maximum heart rate suggests poor fitness. Start with light cardio exercises and gradually increase intensity.',
            'priority': 'Medium'
        })
    
    # Exercise angina recommendations
    if user_data['exercise_angina'] == 1:
        recommendations.append({
            'category': 'Exercise Safety',
            'icon': '⚠️',
            'advice': 'Exercise-induced chest pain detected. Consult a cardiologist before starting any exercise program.',
            'priority': 'High'
        })
    
    # ST depression recommendations
    if user_data['oldpeak'] > 2.0:
        recommendations.append({
            'category': 'Cardiac Health',
            'icon': '💓',
            'advice': 'Significant ST depression detected. This requires immediate medical attention and lifestyle modifications.',
            'priority': 'Critical'
        })
    
    # High-risk prediction recommendations
    if prediction_prob > 0.7:
        recommendations.append({
            'category': 'High Risk Management',
            'icon': '🚨',
            'advice': 'High heart disease risk detected. Schedule immediate consultation with a cardiologist and consider cardiac rehabilitation.',
            'priority': 'Critical'
        })
    
    # General recommendations for all users
    recommendations.extend([
        {
            'category': 'Diet',
            'icon': '🍽️',
            'advice': 'Follow a Mediterranean diet rich in fruits, vegetables, whole grains, and lean proteins.',
            'priority': 'Medium'
        },
        {
            'category': 'Exercise',
            'icon': '💪',
            'advice': 'Aim for 150 minutes of moderate-intensity aerobic activity per week, as recommended by health guidelines.',
            'priority': 'Medium'
        },
        {
            'category': 'Stress Management',
            'icon': '🧘',
            'advice': 'Practice stress-reduction techniques like meditation, deep breathing, or yoga to support heart health.',
            'priority': 'Low'
        }
    ])
    
    return recommendations

# PAGE 1: Heart Disease Prediction
if page == "Heart Disease Prediction":
    st.title("💓 Heart Disease Prediction")
    st.write("Fill in the patient details to predict the likelihood of heart disease.")

    # User inputs for all features
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    chest_pain = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4], 
                             help="1: Typical Angina, 2: Atypical Angina, 3: Non-Anginal Pain, 4: Asymptomatic")
    resting_bp = st.number_input("Resting Blood Pressure (mmHg)", min_value=50, max_value=250, value=120)
    cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], 
                             format_func=lambda x: "Yes" if x == 1 else "No")
    resting_ecg = st.selectbox("Resting ECG", options=[0, 1, 2],
                              help="0: Normal, 1: ST-T Wave Abnormality, 2: Left Ventricular Hypertrophy")
    max_hr = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=150)
    exercise_angina = st.selectbox("Exercise Induced Angina", options=[0, 1], 
                                  format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak = st.number_input(
    "Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1,
    help="How much the heart's ECG line drops below normal. Higher values may mean more heart stress.")
    st_slope = st.selectbox("ST Slope", options=[1, 2, 3],
                           help="1: Upsloping, 2: Flat, 3: Downsloping")

    # Predict button
    if st.button("Predict Heart Disease Risk"):
        # Store user data in session state
        st.session_state.user_data = {
            'age': age,
            'sex': sex,
            'chest_pain': chest_pain,
            'resting_bp': resting_bp,
            'cholesterol': cholesterol,
            'fasting_bs': fasting_bs,
            'resting_ecg': resting_ecg,
            'max_hr': max_hr,
            'exercise_angina': exercise_angina,
            'oldpeak': oldpeak,
            'st_slope': st_slope
        }
        
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
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        # Store prediction results
        st.session_state.prediction_result = {
            'prediction': prediction,
            'probability': probability
        }
        st.session_state.prediction_made = True
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error(f"⚠️ High Risk of Heart Disease")
                st.write(f"**Risk Probability:** {probability:.1%}")
            else:
                st.success(f"✅ Low Risk of Heart Disease")
                st.write(f"**Risk Probability:** {probability:.1%}")
        
        with col2:
            # Risk level indicator
            if probability > 0.7:
                st.write("**Risk Level:** 🔴 **Critical**")
            elif probability > 0.5:
                st.write("**Risk Level:** 🟡 **Moderate**")
            else:
                st.write("**Risk Level:** 🟢 **Low**")
        
        # Prompt to view recommendations
        st.info("📋 **Next Step:** Go to the 'Lifestyle Recommendations' page to see personalized advice based on your results!")

# PAGE 2: Lifestyle Recommendations
elif page == "Lifestyle Recommendations":
    st.title("📋 Personalized Lifestyle Recommendations")
    
    if not st.session_state.prediction_made:
        st.warning("⚠️ Please make a prediction first on the 'Heart Disease Prediction' page to see personalized recommendations.")
        st.stop()
    
    # Display prediction summary
    prediction = st.session_state.prediction_result['prediction']
    probability = st.session_state.prediction_result['probability']
    
    st.write("## 📊 Your Risk Assessment Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if prediction == 1:
            st.metric("Risk Status", "High Risk", delta="⚠️")
        else:
            st.metric("Risk Status", "Low Risk", delta="✅")
    
    with col2:
        st.metric("Risk Probability", f"{probability:.1%}")
    
    with col3:
        if probability > 0.7:
            risk_level = "Critical"
            risk_color = "🔴"
        elif probability > 0.5:
            risk_level = "Moderate"
            risk_color = "🟡"
        else:
            risk_level = "Low"
            risk_color = "🟢"
        st.metric("Risk Level", f"{risk_color} {risk_level}")
    
    st.divider()
    
    # Get personalized recommendations
    recommendations = get_lifestyle_recommendations(st.session_state.user_data, probability)
    
    st.write("## 🎯 Personalized Recommendations")
    st.write("Based on your health profile, here are evidence-based lifestyle recommendations:")
    
    # Group recommendations by priority
    critical_recs = [r for r in recommendations if r['priority'] == 'Critical']
    high_recs = [r for r in recommendations if r['priority'] == 'High']
    medium_recs = [r for r in recommendations if r['priority'] == 'Medium']
    low_recs = [r for r in recommendations if r['priority'] == 'Low']
    
    # Display critical recommendations first
    if critical_recs:
        st.write("### 🚨 Critical Priority")
        for rec in critical_recs:
            with st.expander(f"{rec['icon']} {rec['category']}", expanded=True):
                st.error(rec['advice'])
    
    # Display high priority recommendations
    if high_recs:
        st.write("### ⚠️ High Priority")
        for rec in high_recs:
            with st.expander(f"{rec['icon']} {rec['category']}"):
                st.warning(rec['advice'])
    
    # Display medium priority recommendations
    if medium_recs:
        st.write("### 📋 Medium Priority")
        for rec in medium_recs:
            with st.expander(f"{rec['icon']} {rec['category']}"):
                st.info(rec['advice'])
    
    # Display low priority recommendations
    if low_recs:
        st.write("### 💡 General Health Tips")
        for rec in low_recs:
            with st.expander(f"{rec['icon']} {rec['category']}"):
                st.write(rec['advice'])
    
    # Action plan section
    st.divider()
    st.write("## 📅 Recommended Action Plan")
    
    if probability > 0.7:
        st.error("""
        **Immediate Actions (Within 1 Week):**
        - Schedule appointment with a cardiologist
        - Begin monitoring blood pressure daily
        - Start a heart-healthy diet immediately
        - Avoid strenuous physical activity until cleared by doctor
        """)
    elif probability > 0.5:
        st.warning("""
        **Short-term Actions (Within 1 Month):**
        - Schedule check-up with your primary care physician
        - Begin gradual lifestyle modifications
        - Start monitoring key health metrics
        - Consider joining a cardiac rehabilitation program
        """)
    else:
        st.success("""
        **Preventive Actions (Ongoing):**
        - Maintain current healthy habits
        - Continue regular health screenings
        - Stay active and eat a balanced diet
        - Monitor any changes in symptoms
        """)
    
    # Reset option
    st.divider()
    if st.button("🔄 Make New Prediction"):
        st.session_state.prediction_made = False
        st.session_state.user_data = {}
        st.session_state.prediction_result = {}
        st.rerun()