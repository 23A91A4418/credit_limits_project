import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json

# Load model and feature names
@st.cache_resource
def load_assets():
    model = joblib.load("models/xgb_model.pkl")
    with open("models/feature_names.json", "r") as f:
        feature_names = json.load(f)
    return model, feature_names

model, feature_names = load_assets()

st.set_page_config(page_title="Credit Limit Recommender", page_icon="💳")

st.title("💳 Credit Limit Recommendation System")
st.markdown("""
This dashboard uses an advanced XGBoost model to predict the probability of default 
and recommends an optimal credit limit based on the customer's risk profile.
""")

st.sidebar.header("Enter Customer Details")

limit_bal = st.sidebar.slider("Current Credit Limit (₹)", 10000, 500000, 100000)
age = st.sidebar.slider("Age", 18, 80, 30)
sex = st.sidebar.selectbox("Sex", options=[1, 2], format_func=lambda x: "Male" if x==1 else "Female")
education = st.sidebar.selectbox("Education", options=[1, 2, 3, 4], format_func=lambda x: {1:"Grad School", 2:"University", 3:"High School", 4:"Others"}[x])
marriage = st.sidebar.selectbox("Marital Status", options=[1, 2, 3], format_func=lambda x: {1:"Married", 2:"Single", 3:"Others"}[x])

st.sidebar.subheader("Recent Activity")
pay_0 = st.sidebar.slider("Recent Payment Delay (Months)", -2, 10, 0)
bill_amt = st.sidebar.slider("Latest Bill Amount (₹)", 0, 500000, 20000)
pay_amt = st.sidebar.slider("Latest Payment Amount (₹)", 0, 200000, 5000)

# Create a dictionary for all features, setting defaults for historical months (2-6)
input_dict = {
    'LIMIT_BAL': limit_bal,
    'SEX': sex,
    'EDUCATION': education,
    'MARRIAGE': marriage,
    'AGE': age,
    'PAY_0': pay_0,
    'PAY_2': 0, 'PAY_3': 0, 'PAY_4': 0, 'PAY_5': 0, 'PAY_6': 0,
    'BILL_AMT1': bill_amt,
    'BILL_AMT2': bill_amt * 0.9, 'BILL_AMT3': bill_amt * 0.8, 
    'BILL_AMT4': bill_amt * 0.7, 'BILL_AMT5': bill_amt * 0.6, 'BILL_AMT6': bill_amt * 0.5,
    'PAY_AMT1': pay_amt,
    'PAY_AMT2': pay_amt, 'PAY_AMT3': pay_amt, 
    'PAY_AMT4': pay_amt, 'PAY_AMT5': pay_amt, 'PAY_AMT6': pay_amt,
}

# Engineer the 7 features dynamically
avg_bill = (input_dict['BILL_AMT1'] + input_dict['BILL_AMT2'] + input_dict['BILL_AMT3'] +
            input_dict['BILL_AMT4'] + input_dict['BILL_AMT5'] + input_dict['BILL_AMT6']) / 6

avg_pay = (input_dict['PAY_AMT1'] + input_dict['PAY_AMT2'] + input_dict['PAY_AMT3'] +
           input_dict['PAY_AMT4'] + input_dict['PAY_AMT5'] + input_dict['PAY_AMT6']) / 6

input_dict['avg_bill_amt'] = avg_bill
input_dict['credit_utilization'] = avg_bill / limit_bal if limit_bal > 0 else 0
input_dict['avg_payment'] = avg_pay
input_dict['payment_ratio'] = avg_pay / avg_bill if avg_bill > 0 else 0
input_dict['avg_delay'] = (input_dict['PAY_0'] + input_dict['PAY_2'] + input_dict['PAY_3'] +
                           input_dict['PAY_4'] + input_dict['PAY_5'] + input_dict['PAY_6']) / 6

bills = [input_dict[f'BILL_AMT{i}'] for i in range(1, 7)]
input_dict['spending_std'] = np.std(bills)
input_dict['bill_growth'] = (input_dict['BILL_AMT1'] - input_dict['BILL_AMT6']) / (input_dict['BILL_AMT6'] + 1)

# Ensure the DataFrame has columns in the exact order expected by the model
input_df = pd.DataFrame([input_dict], columns=feature_names)

# Fill NaNs with 0
input_df.fillna(0, inplace=True)

# Predict
prob = model.predict_proba(input_df)[0][1]

# Display Results
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Risk Assessment")
    st.metric(label="Default Probability", value=f"{prob:.1%}")
    
    # Risk category
    if prob < 0.2:
        risk = "Low Risk"
        color = "green"
    elif prob < 0.5:
        risk = "Medium Risk"
        color = "orange"
    else:
        risk = "High Risk"
        color = "red"
        
    st.markdown(f"**Risk Category:** :{color}[**{risk}**]")

with col2:
    st.subheader("💡 Recommendation")
    
    # Recommendation
    if risk == "Low Risk":
        recommended = limit_bal * 1.5
        reason = "Customer shows excellent repayment behavior. Safe to increase limit by 50%."
    elif risk == "Medium Risk":
        recommended = limit_bal
        reason = "Customer shows moderate risk. Maintain current limit."
    else:
        recommended = limit_bal * 0.7
        reason = "Customer is high risk. Reduce limits by 30% to mitigate exposure."
        
    st.metric(label="Recommended Credit Limit", value=f"₹ {int(recommended):,}", delta=int(recommended - limit_bal))
    st.info(reason)

st.divider()

st.subheader("🔍 Feature Explainability (Top Factors)")
st.write("The following indicators were strongly considered for this profile:")
col3, col4, col5 = st.columns(3)
col3.metric("Credit Utilization", f"{input_dict['credit_utilization']:.2f}")
col4.metric("Avg Payment Delay", f"{input_dict['avg_delay']:.1f} months")
col5.metric("Payment Ratio", f"{input_dict['payment_ratio']:.2f}")