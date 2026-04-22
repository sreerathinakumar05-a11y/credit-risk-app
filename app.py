import streamlit as st
import pandas as pd
import joblib

st.title("Credit Risk Prediction App")

# Load model
data = joblib.load("credit_risk_model.pkl")
model = data["model"]
scaler = data["scaler"]
features = data["features"]

# Inputs
LIMIT_BAL = st.number_input("LIMIT_BAL", value=200000)
AGE = st.number_input("AGE", value=30)

PAY_0 = st.number_input("PAY_0", value=0)
PAY_2 = st.number_input("PAY_2", value=0)
PAY_3 = st.number_input("PAY_3", value=0)
PAY_4 = st.number_input("PAY_4", value=0)
PAY_5 = st.number_input("PAY_5", value=0)
PAY_6 = st.number_input("PAY_6", value=0)

BILL_AMT1 = st.number_input("BILL_AMT1", value=50000)
BILL_AMT2 = st.number_input("BILL_AMT2", value=45000)
BILL_AMT3 = st.number_input("BILL_AMT3", value=40000)
BILL_AMT4 = st.number_input("BILL_AMT4", value=35000)
BILL_AMT5 = st.number_input("BILL_AMT5", value=30000)
BILL_AMT6 = st.number_input("BILL_AMT6", value=25000)

PAY_AMT1 = st.number_input("PAY_AMT1", value=5000)
PAY_AMT2 = st.number_input("PAY_AMT2", value=5000)
PAY_AMT3 = st.number_input("PAY_AMT3", value=5000)
PAY_AMT4 = st.number_input("PAY_AMT4", value=5000)
PAY_AMT5 = st.number_input("PAY_AMT5", value=5000)
PAY_AMT6 = st.number_input("PAY_AMT6", value=5000)

# Create dataframe
input_df = pd.DataFrame([{
    "LIMIT_BAL": LIMIT_BAL,
    "AGE": AGE,
    "PAY_0": PAY_0,
    "PAY_2": PAY_2,
    "PAY_3": PAY_3,
    "PAY_4": PAY_4,
    "PAY_5": PAY_5,
    "PAY_6": PAY_6,
    "BILL_AMT1": BILL_AMT1,
    "BILL_AMT2": BILL_AMT2,
    "BILL_AMT3": BILL_AMT3,
    "BILL_AMT4": BILL_AMT4,
    "BILL_AMT5": BILL_AMT5,
    "BILL_AMT6": BILL_AMT6,
    "PAY_AMT1": PAY_AMT1,
    "PAY_AMT2": PAY_AMT2,
    "PAY_AMT3": PAY_AMT3,
    "PAY_AMT4": PAY_AMT4,
    "PAY_AMT5": PAY_AMT5,
    "PAY_AMT6": PAY_AMT6
}])

# Feature Engineering
input_df["total_bill"] = input_df[
    ["BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6"]
].sum(axis=1)

input_df["total_payment"] = input_df[
    ["PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]
].sum(axis=1)

input_df["credit_utilization"] = input_df["total_bill"] / (input_df["LIMIT_BAL"] + 1)

input_df["delay_count"] = (
    (input_df[["PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]] > 0)
).sum(axis=1)

input_df["payment_ratio"] = input_df["total_payment"] / (input_df["total_bill"] + 1)

input_df["early_warning"] = (
    (input_df["delay_count"] >= 2) & (input_df["payment_ratio"] < 0.6)
).astype(int)

# Align columns
input_df = input_df.reindex(columns=features, fill_value=0)

# Prediction
if st.button("Predict"):
    scaled = scaler.transform(input_df)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    st.write("Risk Score:", round(prob, 3))
    st.write("Prediction:", "Default" if pred == 1 else "No Default")