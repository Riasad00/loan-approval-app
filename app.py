import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Loan Approval Predictor", page_icon="💰")
st.title("🏦 Simple Loan Approval System")
st.markdown("Enter the 4 details below to check if the loan will be approved.")

# Load model (cached)
@st.cache_resource
def load_model():
    return joblib.load('simple_loan_model.joblib')

model = load_model()

# User inputs
applicant_income = st.number_input("Applicant's Annual Income", min_value=0, value=5000)
loan_amount = st.number_input("Loan Amount", min_value=0, value=20000)
credit_history = st.selectbox("Credit History", options=[1.0, 0.0], format_func=lambda x: "Good (1.0)" if x==1.0 else "Bad/None (0.0)")
loan_term = st.number_input("Loan Term (in months)", min_value=0, value=360)

# Prediction button
if st.button("✅ Check Loan Approval"):
    input_df = pd.DataFrame({
        'ApplicantIncome': [applicant_income],
        'LoanAmount': [loan_amount],
        'Credit_History': [credit_history],
        'Loan_Amount_Term': [loan_term]
    })
    # Ensure feature order matches training
    input_df = input_df[model.feature_names_in_]
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    
    if prediction == 1:
        st.success(f"🎉 **Loan Approved!** (Confidence: {proba*100:.2f}%)")
        st.balloons()
    else:
        st.error(f"❌ **Loan Rejected** (Approval probability: {proba*100:.2f}%)")

st.markdown("---")
st.caption("⚠️ This is an educational project – do not use for real financial decisions.")