import streamlit as st
import pickle
import numpy as np

# Load saved model
with open("disease_model.pkl", "rb") as f:
    model = pickle.load(f)
st.title("🩺 Disease Prediction System")

fever = st.number_input("Fever", min_value=0, max_value=1,)
headache = st.number_input("Headache", min_value=0, max_value=1)
nausea = st.number_input("Nausea", min_value=0, max_value=1)
vomiting = st.number_input("Vomiting", min_value=0, max_value=1)
fatigue = st.number_input("Fatigue", min_value=0, max_value=1)
joint_pain = st.number_input("Joint Pain", min_value=0, max_value=1,)
skin_rash = st.number_input("Skin Rash", min_value=0, max_value=1,)
cough = st.number_input("Cough", min_value=0, max_value=1)
weight_loss = st.number_input("wieght loss", min_value=0, max_value=1)
yellow_eyes = st.number_input("yellow eyes", min_value=0, max_value=1)

if st.button("Predict"):
    features = np.array([[fever,headache,nausea,vomiting,fatigue,joint_pain,skin_rash,cough,weight_loss,yellow_eyes]])
    prediction = model.predict(features)[0]
    result = "⚠️ Disease Detected" if prediction == 1 else "✅ No Disease Detected"
    st.subheader(f"Result: {result}")



