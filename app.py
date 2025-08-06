# app.py
import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('employee_classifier.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Employee Classifier", layout="centered")

st.title("🧠 Employee Performance Classifier")
st.markdown("Enter the employee details to predict if they are **Good** or **Bad**.")

# Input fields
satisfaction = st.slider("Satisfaction (0 to 1)", 0.0, 1.0, 0.5, step=0.01)
number_project = st.number_input("Number of Projects", min_value=1, max_value=10, value=3)
average_montly_hours = st.number_input("Average Monthly Hours", min_value=50, max_value=400, value=160)
time_spend_company = st.number_input("Years in Company", min_value=0, max_value=20, value=3)

if st.button("🔍 Predict"):
    input_data = {
        'satisfaction': satisfaction,
        'number_project': number_project,
        'average_montly_hours': average_montly_hours,
        'time_spend_company': time_spend_company
    }

    # Convert to DataFrame and scale
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    label = "🟢 Good Employee" if prediction == 1 else "🔴 Bad Employee"

    st.subheader("Prediction:")
    st.success(label if prediction == 1 else label)
