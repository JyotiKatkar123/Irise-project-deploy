import streamlit as st


# This caches the model so it doesn't reload on every button click
@st.cache_resource
def load_model():
    return pipeline('sentiment-analysis')

obj = load_model()

st.title("Sentiment Analyzer")
st.write("Enter text below to see if it's Positive or Negative.")

txt = st.text_input("Input text:", "I like this product")

if st.button("Analyze"):
    re = obj(txt)
    label = re[0]['label']
    score = re[0]['score']
    
    if label == 'POSITIVE':
        st.success(f"Result: {label} (Score: {score:.4f})")
    else:
        st.error(f"Result: {label} (Score: {score:.4f})")
