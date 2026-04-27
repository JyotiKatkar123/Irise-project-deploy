import streamlit as st
import pickle
import numpy as np
import os

# Set page config
st.set_page_config(page_title="Diabetes Prediction System", layout="wide")

# Modern Professional Styling
st.markdown("""
    <style>
    .main-card {
        background-color: #ffffff;
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        max-width: 900px;
        margin: auto;
    }
    .title-text { color: #2c3e50; text-align: center; font-weight: 700; }
    .prediction-line {
        width: 100%;
        text-align: center;
        padding: 15px;
        border-radius: 12px;
        font-size: 22px;
        font-weight: bold;
        margin-top: 25px;
        color: white;
    }
    .stButton > button {
        background-color: #3498db;
        color: white;
        width: 100%;
        height: 50px;
        border-radius: 10px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Corrected Loading Logic
@st.cache_resource
def load_model():
    model_path = 'model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    else:
        return None

model = load_model()

# UI Layout
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown('<h1 class="title-text">Diabetes Prediction System</h1>', unsafe_allow_html=True)

if model is None:
    st.error("Error: 'model.pkl' not found. Please ensure the file is in the same directory.")
else:
    # Grid Layout matching your requirements
    r1_c1, r1_c2, r1_c3 = st.columns(3)
    with r1_c1:
        pregnancies = st.number_input("Pregnancies", min_value=0, step=1, value=0)
    with r1_c2:
        glucose = st.number_input("Glucose", min_value=0.0, value=120.0)
    with r1_c3:
        blood_pressure = st.number_input("Blood Pressure", min_value=0.0, value=70.0)

    r2_c1, r2_c2, r2_c3 = st.columns(3)
    with r2_c1:
        skin_thickness = st.number_input("Skin Thickness", min_value=0.0, value=20.0)
    with r2_c2:
        insulin = st.number_input("Insulin", min_value=0.0, value=80.0)
    with r2_c3:
        bmi = st.number_input("BMI", min_value=0.0, value=25.0)

    r3_spacer1, r3_c1, r3_c2, r3_spacer2 = st.columns([0.1, 1, 1, 0.1])
    with r3_c1:
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f", value=0.471)
    with r3_c2:
        age = st.number_input("Age", min_value=0, step=1, value=33)

    if st.button("PREDICT"):
        # Features must be in exact order
        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                              insulin, bmi, dpf, age]])
        
        prediction = model.predict(features)
        
        if prediction[0] == 1:
            st.markdown('<div class="prediction-line" style="background-color: #e74c3c;">Result: The person is Diabetic</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction-line" style="background-color: #2ecc71;">Result: The person is Not Diabetic</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
