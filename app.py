import streamlit as st
import pickle
import numpy as np
import os

# Set page configuration
st.set_page_config(page_title="Diabetes Prediction System", layout="wide")

# Modern Styling
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

@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as file:
            return pickle.load(file)
    except ModuleNotFoundError:
        return "SKLEARN_MISSING"
    except Exception as e:
        return str(e)

model_result = load_model()

st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown('<h1 style="text-align:center;">Diabetes Prediction System</h1>', unsafe_allow_html=True)

if model_result == "SKLEARN_MISSING":
    st.error("### 🚨 Critical Error: Scikit-Learn Not Found")
    st.info("To fix this: Add **scikit-learn** to your **requirements.txt** file in GitHub and wait for the app to reboot.")
elif isinstance(model_result, str):
    st.error(f"Error loading model: {model_result}")
else:
    # Use the loaded model
    model = model_result
    
    # Input Grid
    r1_c1, r1_c2, r1_c3 = st.columns(3)
    with r1_c1: pregnancies = st.number_input("Pregnancies", value=0)
    with r1_c2: glucose = st.number_input("Glucose", value=120.0)
    with r1_c3: blood_pressure = st.number_input("Blood Pressure", value=70.0)

    r2_c1, r2_c2, r2_c3 = st.columns(3)
    with r2_c1: skin_thickness = st.number_input("Skin Thickness", value=20.0)
    with r2_c2: insulin = st.number_input("Insulin", value=80.0)
    with r2_c3: bmi = st.number_input("BMI", value=25.0)

    r3_spacer, r3_c1, r3_c2, r3_spacer2 = st.columns([0.1, 1, 1, 0.1])
    with r3_c1: dpf = st.number_input("Diabetes Pedigree Function", value=0.471, format="%.3f")
    with r3_c2: age = st.number_input("Age", value=33)

    if st.button("PREDICT"):
        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        prediction = model.predict(features)
        
        if prediction[0] == 1:
            st.markdown('<div class="prediction-line" style="background-color: #e74c3c;">The person is Diabetic</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction-line" style="background-color: #2ecc71;">The person is Not Diabetic</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
