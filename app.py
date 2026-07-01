import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('fatigue_model.pkl')

st.set_page_config(page_title="Steel Fatigue Strength Predictor", layout="centered")
st.title("🔩 Steel Fatigue Strength Predictor")
st.write("Predict fatigue strength from steel composition and heat treatment parameters, using an XGBoost model.")

st.header("Composition (% weight)")
col1, col2, col3 = st.columns(3)
with col1:
    C = st.number_input("Carbon (C)", 0.17, 0.63, 0.39, 0.01)
    Si = st.number_input("Silicon (Si)", 0.16, 2.05, 0.30, 0.01)
    Mn = st.number_input("Manganese (Mn)", 0.37, 1.60, 0.82, 0.01)
with col2:
    P = st.number_input("Phosphorus (P)", 0.002, 0.031, 0.016, 0.001, format="%.3f")
    S = st.number_input("Sulfur (S)", 0.003, 0.030, 0.015, 0.001, format="%.3f")
    Ni = st.number_input("Nickel (Ni)", 0.01, 2.78, 0.52, 0.01)
with col3:
    Cr = st.number_input("Chromium (Cr)", 0.01, 1.17, 0.57, 0.01)
    Cu = st.number_input("Copper (Cu)", 0.01, 0.26, 0.07, 0.01)
    Mo = st.number_input("Molybdenum (Mo)", 0.00, 0.24, 0.07, 0.01)

st.header("Heat Treatment Parameters")
col4, col5, col6 = st.columns(3)
with col4:
    NT = st.number_input("Normalizing Temp (NT)", 825, 930, 872)
    THT = st.number_input("Through Hardening Temp (THT)", 30, 865, 738)
    THt = st.number_input("Through Hardening Time (THt)", 0, 30, 26)
    THQCr = st.number_input("Through Hardening Cooling Rate (THQCr)", 0, 24, 11)
with col5:
    CT = st.number_input("Carburization Temp (CT)", 30, 930, 129)
    Ct = st.number_input("Carburization Time (Ct)", 0, 540, 41)
    DT = st.number_input("Diffusion Temp (DT)", 30, 903, 124)
    Dt = st.number_input("Diffusion Time (Dt)", 0.0, 70.2, 4.8)
with col6:
    QmT = st.number_input("Quenching Media Temp (QmT)", 30, 140, 35)
    TT = st.number_input("Tempering Temp (TT)", 30, 680, 537)
    Tt = st.number_input("Tempering Time (Tt)", 0, 120, 65)
    TCr = st.number_input("Tempering Cooling Rate (TCr)", 0, 24, 21)

st.header("Process Details")
col7, col8 = st.columns(2)
with col7:
    RedRatio = st.number_input("Reduction Ratio", 240, 5530, 924)
    dA = st.number_input("Area Proportion of Inclusions Type A (dA)", 0.0, 0.13, 0.05)
with col8:
    dB = st.number_input("Area Proportion of Inclusions Type B (dB)", 0.0, 0.05, 0.003)
    dC = st.number_input("Area Proportion of Inclusions Type C (dC)", 0.0, 0.058, 0.008)

if st.button("Predict Fatigue Strength", type="primary"):
    input_data = pd.DataFrame([{
        'NT': NT, 'THT': THT, 'THt': THt, 'THQCr': THQCr,
        'CT': CT, 'Ct': Ct, 'DT': DT, 'Dt': Dt,
        'QmT': QmT, 'TT': TT, 'Tt': Tt, 'TCr': TCr,
        'C': C, 'Si': Si, 'Mn': Mn, 'P': P, 'S': S,
        'Ni': Ni, 'Cr': Cr, 'Cu': Cu, 'Mo': Mo,
        'RedRatio': RedRatio, 'dA': dA, 'dB': dB, 'dC': dC
    }])

    prediction = model.predict(input_data)[0]
    st.success(f"### Predicted Fatigue Strength: **{prediction:.2f} MPa**")

st.caption("Model: XGBoost Regressor | R² ≈ 0.99 on test set")
