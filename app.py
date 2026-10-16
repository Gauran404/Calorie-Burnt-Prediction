"""
=============================================================================
CALORIES BURNT PREDICTOR: STREAMLIT FRONTEND WEB APPLICATION
=============================================================================
Frontend Features:
1. User Workout Input & Real-time Calorie Burn Prediction (Model runs on backend)
2. Feature Correlation Heatmap (Seaborn / Matplotlib)
3. Workout Profile Clustering (PCA 2D projection with user session marker)

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
        "pca_kmeans": joblib.load(os.path.join(MODELS_DIR, "pca_kmeans.joblib")),
        "corr_mat": joblib.load(os.path.join(MODELS_DIR, "correlation_matrix.joblib")),
        "pca_sample": pd.read_csv(os.path.join(MODELS_DIR, "pca_sample.csv"))
    }
    return artifacts


artifacts = load_backend_artifacts()
model = artifacts["model"]
scaler = artifacts["scaler"]
pca_kmeans = artifacts["pca_kmeans"]
corr_mat = artifacts["corr_mat"]
pca_sample = artifacts["pca_sample"]


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

st.markdown("---")

# ---------------------------------------------------------------------------
# VISUALIZATIONS SECTION: HEATMAP & CLUSTER (SIDE BY SIDE)
# ---------------------------------------------------------------------------
col_heat, col_clust = st.columns(2)

# --- VISUALIZATION 1: CORRELATION HEATMAP ---
with col_heat:
    st.subheader("Feature Correlation Heatmap")
    st.caption("Correlation between physiological parameters and calories burnt:")

    fig_heat, ax_heat = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        corr_mat,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax_heat,
        linewidths=0.5
    )
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_heat)

# --- VISUALIZATION 2: PCA 2D WORKOUT CLUSTER MAP ---
with col_clust:
    st.subheader("Workout Clusters (PCA 2D)")
    st.caption("Workout profile segmentation with your current session projected live:")

    # Project current user workout onto 2D PCA
    c_scaler = pca_kmeans["scaler"]
    c_pca = pca_kmeans["pca"]
    c_kmeans = pca_kmeans["kmeans"]
    c_personas = pca_kmeans["cluster_personas"]
    c_features = pca_kmeans["cluster_features"]

    user_vec_df = pd.DataFrame([{
        "Duration": float(duration),
        "Heart_Rate": float(heart_rate),
        "Body_Temp": float(body_temp),
        "Calories": float(predicted_calories)
    }])[c_features]
    user_vec_scaled = c_scaler.transform(user_vec_df)
    user_pc = c_pca.transform(user_vec_scaled)[0]

    # Plot Cluster Scatter
    fig_pca, ax_pca = plt.subplots(figsize=(6, 5))
    cluster_colors = {0: "#3498db", 1: "#f39c12", 2: "#e74c3c"}

    for cid in [0, 1, 2]:
        subset = pca_sample[pca_sample["Cluster"] == cid]
        ax_pca.scatter(
            subset["PC1"],
            subset["PC2"],
            c=cluster_colors[cid],
            label=c_personas[cid],
            alpha=0.5,
            s=35,
            edgecolor="none"
        )

    # Plot User's current location as a distinct gold star with black outline
    ax_pca.scatter(
        user_pc[0],
        user_pc[1],
        c="#FFD700",
        marker="*",
        s=320,
        edgecolor="black",
        linewidth=1.8,
        label="Your Workout Session",
        zorder=5
    )

    var_p1 = pca_kmeans["var_exp"][0] * 100
    var_p2 = pca_kmeans["var_exp"][1] * 100
    ax_pca.set_xlabel(f"Principal Component 1 ({var_p1:.1f}% var)", fontsize=9)
    ax_pca.set_ylabel(f"Principal Component 2 ({var_p2:.1f}% var)", fontsize=9)
    ax_pca.legend(fontsize=8, loc="upper left")
    ax_pca.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig_pca)

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center; color:#888; font-size:0.85rem;'>Calories Burnt Prediction System | Powered by Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn & Streamlit</div>", unsafe_allow_html=True)
