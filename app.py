import streamlit as st
import pickle
import numpy as np
import os

# Page Config
st.set_page_config(page_title="Diabetes Prediction System", layout="wide")

# Modern Professional Styling
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
        # Step 1: Check if file exists
        if not os.path.exists('model.pkl'):
            return "FILE_NOT_FOUND"
        
        with open('model.pkl', 'rb') as file:
            return pickle.load(file)
            
    except Exception as e:
        return str(e)

model_result = load_model()

st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown('<h1 style="text-align:center;">Diabetes Prediction System</h1>', unsafe_allow_html=True)

# Logic to handle different types of errors
if model_result == "FILE_NOT_FOUND":
    st.error("🚨 'model.pkl' is missing from your GitHub repository.")
elif isinstance(model_result, str):
    st.error(f"### 🚨 Model Compatibility Error")
    st.write(f"Details: {model_result}")
    st.info("Check if your **requirements.txt** has: **scikit-learn==1.2.2**")
else:
    model = model_result
    
    # Feature Input Grid (Exact values from your previous layout)
    r1_c1, r1_c2, r1_c3 = st.columns(3)
    with r1_c1: preg = st.number_input("Pregnancies", value=0)
    with r1_c2: gluc = st.number_input("Glucose", value=120.0)
    with r1_c3: bp = st.number_input("Blood Pressure", value=70.0)

    r2_c1, r2_c2, r2_c3 = st.columns(3)
    with r2_c1: skin = st.number_input("Skin Thickness", value=20.0)
    with r2_c2: ins = st.number_input("Insulin", value=80.0)
    with r2_c3: bmi = st.number_input("BMI", value=25.0)

    r3_sp, r3_c1, r3_c2, r3_sp2 = st.columns([0.1, 1, 1, 0.1])
    with r3_c1: dpf = st.number_input("Diabetes Pedigree Function", value=0.471, format="%.3f")
    with r3_c2: age = st.number_input("Age", value=33)

    if st.button("PREDICT"):
        # Order: [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]
        input_data = np.array([[preg, gluc, bp, skin, ins, bmi, dpf, age]])
        prediction = model.predict(input_data)
        
        if prediction[0] == 1:
            st.markdown('<div class="prediction-line" style="background-color: #e74c3c;">The person is Diabetic</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction-line" style="background-color: #2ecc71;">The person is Not Diabetic</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
