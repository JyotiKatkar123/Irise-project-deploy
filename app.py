import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# --- Page Configuration ---
st.set_page_config(page_title="Diabetes Prediction System", layout="wide")

# --- Professional Styling ---
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

# --- Model Loading Logic ---
@st.cache_resource
def load_model():
    try:
        # Step 1: Ensure the file exists
        if not os.path.exists('model.pkl'):
            return "FILE_NOT_FOUND"
        
        # Step 2: Load the pickle file
        with open('model.pkl', 'rb') as file:
            return pickle.load(file)
            
    except Exception as e:
        return str(e)

model_result = load_model()

# --- User Interface ---
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown('<h1 style="text-align:center; color: #2c3e50;">🩺 Diabetes Prediction System</h1>', unsafe_allow_html=True)

# Handle Loading Errors
if model_result == "FILE_NOT_FOUND":
    st.error("🚨 **'model.pkl' is missing!** Please rename your model file to exactly 'model.pkl' on GitHub.")
elif isinstance(model_result, str):
    st.error(f"### 🚨 Compatibility Error")
    st.write(f"The model failed to load: {model_result}")
    st.info("Make sure your **requirements.txt** includes the correct scikit-learn version.")
else:
    model = model_result
    
    st.markdown("### Enter Patient Health Metrics")
    
    # Input Layout
    r1_c1, r1_c2, r1_c3 = st.columns(3)
    with r1_c1: preg = st.number_input("Pregnancies", min_value=0, step=1, value=1)
    with r1_c2: gluc = st.number_input("Glucose", min_value=0.0, value=120.0)
    with r1_c3: bp = st.number_input("Blood Pressure", min_value=0.0, value=70.0)

    r2_c1, r2_c2, r2_c3 = st.columns(3)
    with r2_c1: skin = st.number_input("Skin Thickness", min_value=0.0, value=20.0)
    with r2_c2: ins = st.number_input("Insulin", min_value=0.0, value=80.0)
    with r2_c3: bmi = st.number_input("BMI", min_value=0.0, value=25.0)

    # Center the last two inputs
    _, r3_c1, r3_c2, _ = st.columns([0.1, 1, 1, 0.1])
    with r3_c1: dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.47, format="%.3f")
    with r3_c2: age = st.number_input("Age", min_value=1, step=1, value=33)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Prediction Execution ---
    if st.button("RUN PREDICTION"):
        try:
            # Create a DataFrame with the exact column names the model expects
            feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                             'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            
            input_df = pd.DataFrame([[preg, gluc, bp, skin, ins, bmi, dpf, age]], 
                                     columns=feature_names)
            
            # Predict
            prediction = model.predict(input_df)
            
            # Display Result
            if prediction[0] == 1:
                st.markdown('<div class="prediction-line" style="background-color: #e74c3c;">⚠️ The model predicts: DIABETIC</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-line" style="background-color: #2ecc71;">✅ The model predicts: NOT DIABETIC</div>', unsafe_allow_html=True)
            
            # Show Probability if available
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_df)[0][1] * 100
                st.write(f"**Confidence Level:** {prob:.2f}% probability of being diabetic.")

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("This often happens if the input columns don't match the model training data.")

st.markdown('</div>', unsafe_allow_html=True)
