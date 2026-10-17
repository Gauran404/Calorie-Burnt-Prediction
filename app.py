"""
=============================================================================
CALORIES BURNT PREDICTOR: STREAMLIT FRONTEND WEB APPLICATION
=============================================================================
Frontend Features:
1. User Workout Input & Real-time Calorie Burn Prediction (Model runs on backend)
2. Burn Rate (kcal/min) and Exercise Intensity Classification
3. Biometric Body Mass Index (BMI) Computation

Packages: numpy, pandas, scikit-learn, matplotlib, seaborn, streamlit, joblib
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Paths configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Streamlit Page Setup
st.set_page_config(
    page_title="Calories Burnt Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .title-banner {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FF4B4B;
        margin-bottom: 0.2rem;
    }
    .metric-container {
        background-color: #FFF5F5;
        border: 2px solid #FF4B4B;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2.8rem;
        font-weight: 900;
        color: #D9383A;
    }
    .metric-label {
        font-size: 1.05rem;
        font-weight: 600;
        color: #555;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# LOAD TRAINED ARTIFACTS FROM BACKEND (Cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_backend_artifacts():
    model_path = os.path.join(MODELS_DIR, "model.joblib")
    if not os.path.exists(model_path):
        import subprocess
        subprocess.run([sys.executable, os.path.join(BASE_DIR, "train_models.py")], check=True)

    artifacts = {
        "model": joblib.load(os.path.join(MODELS_DIR, "model.joblib")),
        "scaler": joblib.load(os.path.join(MODELS_DIR, "scaler.joblib")),
    }
    return artifacts


artifacts = load_backend_artifacts()
model = artifacts["model"]
scaler = artifacts["scaler"]


# ---------------------------------------------------------------------------
# SIDEBAR: USER PHYSICAL & EXERCISE INPUT BOXES (NO SLIDERS / SCROLLBARS)
# ---------------------------------------------------------------------------
st.sidebar.markdown("## Workout Session Inputs")
st.sidebar.caption("Enter your biometric and workout details:")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"], index=0)
gender_val = 0 if gender == "Male" else 1

age = st.sidebar.number_input("Age (Years)", min_value=10, max_value=100, value=28, step=1)
height = st.sidebar.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=175.0, step=0.5)
weight = st.sidebar.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=72.0, step=0.5)

# BMI calculation
height_m = height / 100.0
bmi = round(weight / (height_m ** 2), 1)

st.sidebar.markdown("---")
st.sidebar.markdown("### Workout Activity")
duration = st.sidebar.number_input("Duration (Minutes)", min_value=1.0, max_value=120.0, value=25.0, step=0.5)
heart_rate = st.sidebar.number_input("Heart Rate (BPM)", min_value=40.0, max_value=220.0, value=125.0, step=1.0)
body_temp = st.sidebar.number_input("Body Temperature (°C)", min_value=35.0, max_value=44.0, value=39.5, step=0.1)


# ---------------------------------------------------------------------------
# BACKEND PREDICTION EXECUTION
# ---------------------------------------------------------------------------
user_df = pd.DataFrame([{
    "Gender": gender_val,
    "Age": int(age),
    "Height": float(height),
    "Weight": float(weight),
    "Duration": float(duration),
    "Heart_Rate": float(heart_rate),
    "Body_Temp": float(body_temp)
}])

# Predict using backend model
predicted_calories = max(0.0, float(model.predict(user_df)[0]))
predicted_calories = round(predicted_calories, 1)
burn_rate = round(predicted_calories / max(duration, 0.1), 1)

# Intensity Tier
if predicted_calories < 50:
    intensity_label, intensity_color = "Low Intensity", "#3498db"
elif predicted_calories <= 150:
    intensity_label, intensity_color = "Moderate Cardio", "#f39c12"
else:
    intensity_label, intensity_color = "High Intensity Effort", "#e74c3c"


# ---------------------------------------------------------------------------
# MAIN PAGE: HEADER & ESTIMATED CALORIES
# ---------------------------------------------------------------------------
st.markdown('<div class="title-banner">Calories Burnt Prediction System</div>', unsafe_allow_html=True)
st.caption("Machine Learning Workout Energy Expenditure Estimation | Interactive Telemetry Dashboard")

# Prediction Cards Row
col_cal, col_rate, col_bmi = st.columns([1.5, 1, 1])

with col_cal:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value">{predicted_calories} kcal</div>
        <div class="metric-label">Estimated Calories Burnt</div>
    </div>
    """, unsafe_allow_html=True)

with col_rate:
    st.markdown(f"""
    <div style="background-color:#F8F9FA; border-radius:12px; padding:18px; text-align:center; border:1px solid #E0E0E0;">
        <div style="font-size:2rem; font-weight:800; color:#2C3E50;">{burn_rate}</div>
        <div style="font-size:0.9rem; font-weight:600; color:#7F8C8D;">kcal / Minute</div>
        <div style="margin-top:8px; font-size:0.85rem; font-weight:700; color:{intensity_color};">{intensity_label}</div>
    </div>
    """, unsafe_allow_html=True)

with col_bmi:
    st.markdown(f"""
    <div style="background-color:#F8F9FA; border-radius:12px; padding:18px; text-align:center; border:1px solid #E0E0E0;">
        <div style="font-size:2rem; font-weight:800; color:#2C3E50;">{bmi}</div>
        <div style="font-size:0.9rem; font-weight:600; color:#7F8C8D;">Body Mass Index (BMI)</div>
        <div style="margin-top:8px; font-size:0.85rem; font-weight:600; color:#555;">Weight: {weight}kg | Height: {height}cm</div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center; color:#888; font-size:0.85rem;'>Calories Burnt Prediction System | Powered by Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn & Streamlit</div>", unsafe_allow_html=True)
