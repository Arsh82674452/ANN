import streamlit as st
import numpy as np
import joblib
import os
from sklearn.neural_network import MLPClassifier

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Churn Predictor", layout="wide")

st.title("🏦 Customer Churn Prediction using ANN")
st.markdown("Predict whether a customer will churn based on their profile.")

# =========================
# LOAD MODEL SAFELY
# =========================
@st.cache_resource
def load_model_files():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.neural_network import MLPClassifier

    df = pd.read_csv("data/Artificial_Neural_Network_Case_Study_data.csv")
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

    le = LabelEncoder()
    df["Gender"] = le.fit_transform(df["Gender"])
    df = pd.get_dummies(df, columns=["Geography"], drop_first=True)

    X = df.drop(columns=["Exited"]).values
    y = df["Exited"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100)
    model.fit(X_train, y_train)

    return model, scaler

# =========================
# INPUT UI
# =========================
st.subheader("📋 Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    credit_score = st.slider("Credit Score", 300, 850, 600)
    age = st.slider("Age", 18, 92, 40)
    tenure = st.slider("Tenure (Years)", 0, 10, 3)
    balance = st.number_input("Balance", 0.0, 250000.0, 60000.0)

with col2:
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_card = st.selectbox("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x else "No")
    is_active = st.selectbox("Is Active Member?", [1, 0], format_func=lambda x: "Yes" if x else "No")
    salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

with col3:
    gender = st.selectbox("Gender", ["Male", "Female"])
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

# =========================
# ENCODING
# =========================
gender_enc = 1 if gender == "Male" else 0
geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0

# =========================
# PREDICTION
# =========================
st.markdown("---")

if st.button("🚀 Predict Churn"):
    sample = np.array([[credit_score, gender_enc, age, tenure,
                        balance, num_products, has_card,
                        is_active, salary, geo_germany, geo_spain]])

    sample_scaled = scaler.transform(sample)
    prob = model.predict(sample_scaled)[0][0]

    st.subheader("🔍 Prediction Result")

    if prob > 0.5:
        st.error(f"⚠️ Likely to Churn\n\nProbability: **{prob*100:.2f}%**")
    else:
        st.success(f"✅ Likely to Stay\n\nProbability: **{prob*100:.2f}%**")

    st.progress(float(prob))

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("Built with ❤️ using ANN + Streamlit")