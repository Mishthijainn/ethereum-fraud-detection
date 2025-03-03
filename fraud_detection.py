import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("Ethereum Fraud Detection 🚀")

st.subheader("Enter Transaction Details:")

feature_names = [
    "Avg_min_between_sent_tnx", "Avg_min_between_received_tnx",
    "Time_Diff_between_first_and_last (Mins)", "Sent_tnx", "Received_Tnx",
    "Number_of_Created_Contracts", "Unique_Received_From_Addresses",
    "Unique_Sent_To_Addresses", "min_value_received", "max_value_received",
    "avg_val_received", "min_val_sent", "max_val_sent", "avg_val_sent",
    "min_value_sent_to_contract", "max_val_sent_to_contract",
    "avg_value_sent_to_contract",
    "total_transactions (including_tnx_to_create_contract",
    "total_Ether_sent", "total_ether_received",
    "total_ether_sent_contracts", "total_ether_balance", "Total_ERC20_tnxs",
    "ERC20_total_Ether_received", "ERC20_total_ether_sent",
    "ERC20_total_Ether_sent_contract", "ERC20_uniq_sent_addr",
    "ERC20_uniq_rec_addr", "ERC20_uniq_sent_addr.1",
    "ERC20_uniq_rec_contract_addr", "ERC20_min_val_rec",
    "ERC20_max_val_rec", "ERC20_avg_val_rec", "ERC20_min_val_sent",
    "ERC20_max_val_sent", "ERC20_avg_val_sent",
    "ERC20_uniq_sent_token_name", "ERC20_uniq_rec_token_name",
    "ERC20_most_sent_token_type", "ERC20_most_rec_token_type"
]

user_inputs = []
for feature in feature_names:
    user_inputs.append(st.number_input(feature, value=0.0, format="%.5f"))

input_data = pd.DataFrame([user_inputs], columns=feature_names)

if st.button("Check for Fraud"):
    prediction = model.predict(input_data)
    result = "🚨 Fraud Detected!" if prediction[0] == -1 else "✅ Normal Transaction"
    st.subheader(f"Prediction: {result}")

   
    explainer = shap.Explainer(model)
    shap_values = explainer(input_data)

    st.subheader("Feature Importance (SHAP Explanation)")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, input_data, feature_names=feature_names, show=False)
    st.pyplot(fig)
