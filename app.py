import streamlit as st
import pickle
import numpy as np
import pandas as pd  # Added missing pandas import
import os

# --- Page Config ---
st.set_page_config(page_title="Diabetes Prediction System", layout="wide")

# --- Modern Professional Styling ---
st.markdown("""
    <style>
    .stApp { background-color: #f4f7f6; }
    .main-card {
        background-color: #ffffff;
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        max-width: 900px;
        margin: auto;
    }
    .prediction-line {
        width: 100%; text-align: center; padding: 15px;
        border-radius: 12px; font-size: 22px; font-weight: bold;
        margin-top: 25px; color: white;
    }
    .stButton > button {
        background-color: #3498db; color: white;
        width: 100%; height: 50px; border-radius: 10px; font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_model():
    try:
        # Check if file exists in the repository
        if not os.path.exists('model.pkl'):
            return "FILE_NOT_FOUND"
        
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
            
    except Exception as e:
        return str(e)

model_result = load_model()

# --- UI Layout ---
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown('<h1 style="text-align:center; color:#2c3e50;">🩺 Diabetes Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#7f8c8d;">Enter patient metrics below for an instant health assessment.</p>', unsafe_allow_html=True)

# Error Handling
if model_result == "FILE_NOT_FOUND":
    st.error("🚨 **'model.pkl' is missing.** Please upload your trained model file to the GitHub repository.")
elif isinstance(model_result, str):
    st.error(f"### 🚨 Model Compatibility Error")
    st.write(f"Details: {model_result}")
    st.info("Tip: Ensure your **requirements.txt** has: **scikit-learn==1.2.2** (or the version used to train).")
else:
    model = model_result
    
    # Feature Input Grid
    st.markdown("### Patient Metrics")
    r1_c1, r1_c2, r1_c3 = st.columns(3)
    with r1_c1: preg = st.number_input("Pregnancies", min_value=0, step=1, value=0)
    with r1_c2: gluc = st.number_input("Glucose Level", min_value=0.0, value=120.0)
    with r1_c3: bp = st.number_input("Blood Pressure", min_value=0.0, value=70.0)

    r2_c1, r2_c2, r2_c3 = st.columns(3)
    with r2_c1: skin = st.number_input("Skin Thickness", min_value=0.0, value=20.0)
    with r2_c2: ins = st.number_input("Insulin Level", min_value=0.0, value=80.0)
    with r2_c3: bmi = st.number_input("BMI", min_value=0.0, value=25.0)

    r3_sp, r3_c1, r3_c2, r3_sp2 = st.columns([0.1, 1, 1, 0.1])
    with r3_c1: dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.471, format="%.3f")
    with r3_c2: age = st.number_input("Age", min_value=1, step=1, value=33)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("RUN DIAGNOSTIC PREDICTION"):
        try:
            # Using a DataFrame is safer as it maintains feature names
            # Ensure these column names match your training dataset exactly
            feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                             'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            
            input_data = pd.DataFrame([[preg, gluc, bp, skin, ins, bmi, dpf, age]], 
                                     columns=feature_names)
            
            prediction = model.predict(input_data)
            
            # If your model has predict_proba, let's show confidence
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_data)[0][1] * 100
                st.write(f"**Confidence Level:** {proba:.1f}%")

            if prediction[0] == 1:
                st.markdown('<div class="prediction-line" style="background-color: #e74c3c;">⚠️ The person is predicted to be Diabetic</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-line" style="background-color: #2ecc71;">✅ The person is predicted to be NOT Diabetic</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")

st.markdown('</div>', unsafe_allow_html=True)
