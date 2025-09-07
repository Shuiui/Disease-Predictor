import streamlit as st
import pickle
import numpy as np

# Load saved model
model = pickle.load(open("disease_model.pkl", "rb"))

st.title("🩺 Disease Prediction System")

Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
Glucose = st.number_input("Glucose Level", min_value=0, max_value=300)
BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=200)
SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100)
Insulin = st.number_input("Insulin Level", min_value=0, max_value=900)
BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, format="%.1f")
DPF = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, format="%.3f")
Age = st.number_input("Age", min_value=1, max_value=120)

if st.button("Predict"):
    features = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]])
    prediction = model.predict(features)[0]
    result = "⚠️ Disease Detected" if prediction == 1 else "✅ No Disease Detected"
    st.subheader(f"Result: {result}")
