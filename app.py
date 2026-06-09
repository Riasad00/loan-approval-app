import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

MODEL_PATH = 'simple_loan_model.joblib'

# মডেল না থাকলে ট্রেন করুন
if not os.path.exists(MODEL_PATH):
    with st.spinner("Training model for the first time. Please wait..."):
        # ডেটাসেট ডাউনলোড (GitHub থেকে raw CSV)
        url = "https://raw.githubusercontent.com/your-repo/loan-approval-dataset/main/train.csv"
        # অথবা Streamlit Cloud-এ ডেটা আপলোড করে তার পাথ দিন
        # এখানে উদাহরণ হিসেবে আমি ধরে নিচ্ছি dataset.csv ফাইলটি একই ফোল্ডারে আছে
        df = pd.read_csv('train_u6lujuX_CVtuZ9i (1).csv')
        df['Loan_Status'] = df['Loan_Status'].map({'Y':1,'N':0})
        df['Credit_History'].fillna(df['Credit_History'].median(), inplace=True)
        df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
        df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median(), inplace=True)
        X = df[['ApplicantIncome','LoanAmount','Credit_History','Loan_Amount_Term']]
        y = df['Loan_Status']
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        joblib.dump(model, MODEL_PATH)

# মডেল লোড
model = joblib.load(MODEL_PATH)

# Streamlit UI
st.title("Loan Approval Predictor")
applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
loan_amount = st.number_input("Loan Amount", min_value=0, value=20000)
credit_history = st.selectbox("Credit History", [1.0, 0.0], format_func=lambda x: "Good" if x==1.0 else "Bad")
loan_term = st.number_input("Loan Term (months)", min_value=0, value=360)

if st.button("Predict"):
    input_df = pd.DataFrame([[applicant_income, loan_amount, credit_history, loan_term]],
                            columns=['ApplicantIncome','LoanAmount','Credit_History','Loan_Amount_Term'])
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    if pred == 1:
        st.success(f"Approved with {prob*100:.2f}% confidence")
    else:
        st.error(f"Rejected (approval prob {prob*100:.2f}%)")
