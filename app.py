import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('model/fraud_model.pkl')

st.title("Healthcare Insurance Claim Fraud Detection")

st.write("Enter claim details:")

age = st.number_input("Age", min_value=0, max_value=120, value=30)
claim_amount = st.number_input("Claim Amount", min_value=0, value=200000)
hospital_days = st.number_input("Hospital Days", min_value=0, value=3)

if st.button("Predict"):
    input_df = pd.DataFrame([[age, claim_amount, hospital_days]],
                            columns=['age', 'claim_amount', 'hospital_days'])
    prediction = model.predict(input_df)[0]
    
    if prediction == 1:
        st.error("Fraud Claim ❌")
    else:
        st.success("Genuine Claim ✅")
