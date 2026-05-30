import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="centered")

st.markdown("""
<style>
.block-container {
    max-width: 700px;
    padding-top: 2rem;
}
.stButton>button {
    width: 100%;
    border-radius: 6px;
    font-size: 15px;
}
</style>
""", unsafe_allow_html=True)

st.title("🏦 Loan Approval Prediction")
st.write("Enter applicant details to predict whether the loan will be **Approved** or **Rejected**.")

pipe = joblib.load("model/loan_pipeline.pkl")

st.subheader("Applicant Details")

col1, col2 = st.columns(2)

with col1:
    no_of_dependents = st.number_input("No. of Dependents", min_value=0, max_value=20, value=0, step=1)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

with col2:
    income_annum = st.number_input("Annual Income", min_value=0, value=500000, step=10000)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=200000, step=10000)
    loan_term = st.number_input("Loan Term (years)", min_value=1, max_value=40, value=10, step=1)
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=750, step=1)

input_df = pd.DataFrame([{
    "no_of_dependents": no_of_dependents,
    "education": education,
    "self_employed": self_employed,
    "income_annum": income_annum,
    "loan_amount": loan_amount,
    "loan_term": loan_term,
    "cibil_score": cibil_score,
}])

st.divider()

if st.button("Predict"):
    pred = pipe.predict(input_df)[0]
    proba = pipe.predict_proba(input_df)[0]

    if pred == 1:
        st.success(f"Approved (Confidence: {proba[1]*100:.2f}%)")
    else:
        st.error(f"Rejected (Confidence: {proba[0]*100:.2f}%)")

    st.caption("Note: Confidence is based on model probability, not a guarantee.")
    