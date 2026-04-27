import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Title
st.title("Machine Learning Model App")

st.write("Enter input values to get prediction")

# Example inputs (modify based on your model)
feature1 = st.number_input("Enter Feature 1")
feature2 = st.number_input("Enter Feature 2")
feature3 = st.number_input("Enter Feature 3")

# Predict button
if st.button("Predict"):
    # Convert input into array
    input_data = np.array([[feature1, feature2, feature3]])
    
    # Prediction
    prediction = model.predict(input_data)
    
    st.success(f"Prediction: {prediction[0]}")
