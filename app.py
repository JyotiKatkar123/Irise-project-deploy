import streamlit as st
import pickle
import numpy as np

# Page Configuration
st.set_page_config(page_title="Diabetes Prediction System", layout="wide")

# Professional & Creative CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #f4f7f6;
    }
    .main-card {
        background-color: #ffffff;
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        max-width: 900px;
        margin: auto;
    }
    .title-text {
        color: #2c3e50;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .subtitle-text {
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 30px;
    }
    /* Straight-line prediction styling */
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
        font-size: 18px;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #2980b9;
        color: white;
    }
    footer {
        text-align: center;
        color: #95a5a6;
        margin-top: 50px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# App Content
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown('<h1 class="title-text">Diabetes Prediction using ML</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Enter the details below to analyze health status</p>', unsafe_allow_html=True)

# Grid Layout matching the uploaded image exactly
# Row 1
r1_c1, r1_c2, r1_c3 = st.columns(3)
with r1_c1:
    pregnancies = st.number_input("Pregnancies", min_value=0, step=1, value=0)
with r1_c2:
    glucose = st.number_input("Glucose", min_value=0.0, value=120.0)
with r1_c3:
    blood_pressure = st.number_input("Blood Pressure", min_value=0.0, value=70.0)

# Row 2
r2_c1, r2_c2, r2_c3 = st.columns(3)
with r2_c1:
    skin_thickness = st.number_input("Skin Thickness", min_value=0.0, value=20.0)
with r2_c2:
    insulin = st.number_input("Insulin", min_value=0.0, value=80.0)
with r2_c3:
    bmi = st.number_input("BMI", min_value=0.0, value=25.0)

# Row 3 (DPF and Age)
r3_spacer1, r3_c1, r3_c2, r3_spacer2 = st.columns([0.1, 1, 1, 0.1])
with r3_c1:
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f", value=0.471)
with r3_c2:
    age = st.number_input("Age", min_value=0, step=1, value=33)

st.markdown("<br>", unsafe_allow_html=True)

# Predict Button
if st.button("PREDICT"):
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                          insulin, bmi, dpf, age]])
    
    prediction = model.predict(features)
    
    # Logic for Straight Line Prediction Result
    if prediction[0] == 1:
        st.markdown('<div class="prediction-line" style="background-color: #e74c3c;">The person is Diabetic</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="prediction-line" style="background-color: #2ecc71;">The person is Not Diabetic</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Professional Footer
st.markdown("<footer>Data Analysis Project © 2024</footer>", unsafe_allow_html=True)
