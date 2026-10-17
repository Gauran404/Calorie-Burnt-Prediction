"""
=============================================================================
CALORIES BURNT PREDICTOR & HEALTH INTELLIGENCE SYSTEM
=============================================================================
Frontend Modules:
1. Workout Telemetry & Cardiovascular Zone Classification
2. ML-Powered Exercise Ranking & Personalized Rule-Based Recommender
3. Clinical Diet & Nutritional Planning (BMR, TDEE, Calorie Target, Macros)
4. Lifestyle Risk Factor Screening & Preventive Guidelines

Core ML Models: Random Forest Regressor (Ensemble Bagging) from Scikit-Learn
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
    page_title="Calories Burnt Prediction & Health Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .title-banner {
        font-size: 2.1rem;
        font-weight: 800;
        color: #FF4B4B;
        margin-bottom: 0.1rem;
    }
    .metric-container {
        background-color: #FFF5F5;
        border: 2px solid #FF4B4B;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 2.6rem;
        font-weight: 900;
        color: #D9383A;
    }
    .metric-label {
        font-size: 0.95rem;
        font-weight: 600;
        color: #555;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .rec-box {
        background-color: #F0F7FF;
        border-left: 5px solid #2B6CB0;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 18px;
        font-size: 1.05rem;
        color: #1A365D;
        line-height: 1.5;
    }
    .risk-card-normal {
        background-color: #F0FFF4;
        border-left: 4px solid #38A169;
        padding: 14px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .risk-card-warning {
        background-color: #FFFAF0;
        border-left: 4px solid #DD6B20;
        padding: 14px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .risk-card-danger {
        background-color: #FFF5F5;
        border-left: 4px solid #E53E3E;
        padding: 14px;
        border-radius: 8px;
        margin-bottom: 12px;
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
st.sidebar.markdown("## User Profile & Telemetry")
st.sidebar.caption("Enter your biometric and workout session parameters:")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"], index=0)
gender_val = 0 if gender == "Male" else 1

age = st.sidebar.number_input("Age (Years)", min_value=10, max_value=100, value=28, step=1)
height = st.sidebar.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=175.0, step=0.5)
weight = st.sidebar.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=72.0, step=0.5)

fitness_goal = st.sidebar.selectbox(
    "Primary Fitness Objective",
    ["Fat Loss / Weight Management", "Cardiovascular Conditioning & Health", "Muscle Hypertrophy & Athletic Endurance"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Workout Activity Telemetry")
duration = st.sidebar.number_input("Workout Duration (Minutes)", min_value=1.0, max_value=120.0, value=25.0, step=0.5)
heart_rate = st.sidebar.number_input("Heart Rate (BPM)", min_value=40.0, max_value=220.0, value=125.0, step=1.0)
body_temp = st.sidebar.number_input("Body Temperature (°C)", min_value=35.0, max_value=44.0, value=39.5, step=0.1)


# ---------------------------------------------------------------------------
# CORE COMPUTATIONS: BMI, HR ZONES, ML PREDICTIONS
# ---------------------------------------------------------------------------
# 1. BMI Calculation
height_m = height / 100.0
bmi = round(weight / (height_m ** 2), 1)
if bmi < 18.5:
    bmi_cat = "Underweight"
elif bmi < 25.0:
    bmi_cat = "Normal Weight"
elif bmi < 30.0:
    bmi_cat = "Overweight"
else:
    bmi_cat = "Obese"

# 2. Cardiovascular Zone & Workout Type Classification
hr_max = int(220 - age)
hr_pct = round((heart_rate / hr_max) * 100.0, 1)

if hr_pct < 60:
    hr_zone_name = "Zone 1: Active Recovery (< 60% Max HR)"
    workout_type_desc = "Active Recovery & Mobility Workout"
    zone_color = "#3498db"
elif hr_pct < 70:
    hr_zone_name = "Zone 2: Aerobic Base & Fat Oxidation (60-70% Max HR)"
    workout_type_desc = "Steady-State Aerobic Fat-Burning Session"
    zone_color = "#2ecc71"
elif hr_pct < 80:
    hr_zone_name = "Zone 3: Aerobic Tempo & Conditioning (70-80% Max HR)"
    workout_type_desc = "Cardiovascular Aerobic Tempo Session"
    zone_color = "#f39c12"
elif hr_pct < 90:
    hr_zone_name = "Zone 4: Anaerobic Threshold (80-90% Max HR)"
    workout_type_desc = "High-Intensity Anaerobic Threshold Training"
    zone_color = "#e67e22"
else:
    hr_zone_name = "Zone 5: Neuromuscular Maximal Effort (> 90% Max HR)"
    workout_type_desc = "Maximum Effort Sprint & Anaerobic Conditioning"
    zone_color = "#e74c3c"

if duration < 15:
    dur_label = "Short Micro-Burst"
elif duration <= 45:
    dur_label = "Standard Training Volume"
else:
    dur_label = "Prolonged Endurance Volume"

full_workout_classification = f"{workout_type_desc} ({dur_label})"

# 3. Model Prediction on Current Telemetry
user_df = pd.DataFrame([{
    "Gender": gender_val,
    "Age": int(age),
    "Height": float(height),
    "Weight": float(weight),
    "Duration": float(duration),
    "Heart_Rate": float(heart_rate),
    "Body_Temp": float(body_temp)
}])

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
# MAIN PAGE: HEADER & TOP METRICS
# ---------------------------------------------------------------------------
st.markdown('<div class="title-banner">Calories Burnt Prediction & Health Intelligence System</div>', unsafe_allow_html=True)
st.caption("Machine Learning Energy Expenditure Estimation, Cardiovascular Profiling, ML Exercise Ranking & Diet Planning")

# 4 Key Metric Cards
col_cal, col_rate, col_bmi, col_type = st.columns([1.3, 1, 1, 1.2])

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
        <div style="margin-top:6px; font-size:0.85rem; font-weight:700; color:{intensity_color};">{intensity_label}</div>
    </div>
    """, unsafe_allow_html=True)

with col_bmi:
    st.markdown(f"""
    <div style="background-color:#F8F9FA; border-radius:12px; padding:18px; text-align:center; border:1px solid #E0E0E0;">
        <div style="font-size:2rem; font-weight:800; color:#2C3E50;">{bmi}</div>
        <div style="font-size:0.9rem; font-weight:600; color:#7F8C8D;">BMI ({bmi_cat})</div>
        <div style="margin-top:6px; font-size:0.85rem; font-weight:600; color:#555;">{weight}kg | {height}cm</div>
    </div>
    """, unsafe_allow_html=True)

with col_type:
    st.markdown(f"""
    <div style="background-color:#F8F9FA; border-radius:12px; padding:18px; text-align:center; border:1px solid #E0E0E0;">
        <div style="font-size:1.4rem; font-weight:800; color:{zone_color};">{hr_pct}% HR Max</div>
        <div style="font-size:0.9rem; font-weight:600; color:#7F8C8D;">Workout Type</div>
        <div style="margin-top:6px; font-size:0.82rem; font-weight:700; color:#333;">{workout_type_desc}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# ---------------------------------------------------------------------------
# TABS INTERFACE: 4 INTEGRATED SYSTEM MODULES
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Workout Telemetry & Analysis",
    "ML-Ranked Recommendations",
    "Diet & Nutrition Plan",
    "Lifestyle Risk Screening"
])


# ===========================================================================
# TAB 1: WORKOUT TELEMETRY & ANALYSIS
# ===========================================================================
with tab1:
    st.subheader("Cardiovascular Telemetry & Workout Profiling")
    st.caption("Deep physiological analysis derived from your recorded Heart Rate, Workout Duration, and Body Temperature:")

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown(f"""
        **Session Classification:**
        - **Workout Classification:** {full_workout_classification}
        - **Cardiovascular Intensity:** {hr_zone_name}
        - **Session Duration:** {duration} minutes ({dur_label})
        - **Mean Heart Rate:** {heart_rate} BPM (Age-predicted max: {hr_max} BPM)
        - **Relative Cardiovascular Strain:** {hr_pct}% of theoretical maximum
        """)

    with col_t2:
        # Body Temperature Analysis
        if body_temp > 40.0:
            temp_status = "Elevated thermal load (hyperthermic risk)"
        elif body_temp >= 38.5:
            temp_status = "Normal exercise-induced thermogenesis"
        else:
            temp_status = "Resting / low metabolic baseline temperature"

        st.markdown(f"""
        **Thermal & Energy Metrics:**
        - **Core Body Temperature:** {body_temp}°C ({temp_status})
        - **Total Energy Expended:** {predicted_calories} kcal
        - **Caloric Burn Velocity:** {burn_rate} kcal/minute
        - **Primary Energy Substrate:** {"Lipids (Fat Oxidation)" if hr_pct < 70 else "Glycogen (Carbohydrate Metabolism)"}
        """)

    st.markdown("#### Physiological Cardiovascular Zones Reference")
    zones_data = [
        {"Zone": "Zone 1", "HR Range (% Max)": "< 60%", "Intensity": "Very Light", "Primary Benefit": "Active recovery, metabolic waste clearance, joint mobility"},
        {"Zone": "Zone 2", "HR Range (% Max)": "60% - 70%", "Intensity": "Light to Moderate", "Primary Benefit": "Mitochondrial density, maximal fat oxidation, aerobic base"},
        {"Zone": "Zone 3", "HR Range (% Max)": "70% - 80%", "Intensity": "Moderate", "Primary Benefit": "Cardiovascular stamina, capillarization, aerobic tempo"},
        {"Zone": "Zone 4", "HR Range (% Max)": "80% - 90%", "Intensity": "Hard", "Primary Benefit": "Lactate threshold enhancement, high-velocity calorie burn"},
        {"Zone": "Zone 5", "HR Range (% Max)": "> 90%", "Intensity": "Maximum", "Primary Benefit": "Peak VO2 max, neuromuscular speed, sprint power"}
    ]
    st.dataframe(pd.DataFrame(zones_data), hide_index=True, use_container_width=True)


# ===========================================================================
# TAB 2: ML-POWERED EXERCISE RECOMMENDATIONS
# ===========================================================================
with tab2:
    st.subheader("Personalized Exercise Recommendations & ML Ranking Engine")

    # 1. Dynamic Rule-Based Recommendation
    if bmi >= 25.0:
        if bmi >= 30.0:
            rec_activity = "low-impact cycling or incline walking"
            rec_dur = 45
            rec_freq = 4
            rec_target = 1800
            rec_rationale = "shield knee and hip joints while maintaining sustained aerobic fat oxidation"
        else:
            rec_activity = "brisk running or rowing"
            rec_dur = 45
            rec_freq = 4
            rec_target = 1600
            rec_rationale = "accelerate daily caloric expenditure and stimulate metabolic lipid clearance"
    elif bmi < 18.5:
        rec_activity = "moderate bodyweight circuits and swimming"
        rec_dur = 30
        rec_freq = 3
        rec_target = 900
        rec_rationale = "prevent lean muscle loss and avoid excessive caloric depletion"
    else:
        rec_activity = "interval running and functional circuits"
        rec_dur = 40
        rec_freq = 4
        rec_target = 1400
        rec_rationale = "optimize cardiovascular conditioning, metabolic agility, and mitochondrial density"

    rule_recommendation = (
        f"Your BMI is {bmi} ({bmi_cat}) and you burn {predicted_calories} kcal in {duration:.0f} min of your current session, "
        f"so try {rec_dur} min of {rec_activity} {rec_freq} days a week to {rec_rationale} "
        f"(Weekly Target: {rec_target} kcal)."
    )

    st.markdown(f"""
    <div class="rec-box">
        <strong>Personalized Prescription:</strong><br>
        {rule_recommendation}
    </div>
    """, unsafe_allow_html=True)

    # 2. Machine Learning Exercise Ranking
    st.markdown("#### Exercise Suggestions Ranked by Calories Burnt Per Minute")
    st.caption(
        f"Every activity below is evaluated for your exact biometric profile "
        f"(Age: {age}, Gender: {gender}, Weight: {weight}kg, Height: {height}cm) "
        f"using our trained Machine Learning Random Forest Regressor on backend:"
    )

    exercise_catalog = [
        {"Activity": "Outdoor Running", "Duration": 30.0, "Heart_Rate": 155.0, "Body_Temp": 39.8, "Focus": "High-intensity aerobic conditioning"},
        {"Activity": "HIIT Circuit Training", "Duration": 25.0, "Heart_Rate": 165.0, "Body_Temp": 40.0, "Focus": "Maximal anaerobic threshold and afterburn"},
        {"Activity": "Fast Cycling / Spin", "Duration": 30.0, "Heart_Rate": 135.0, "Body_Temp": 39.2, "Focus": "Cardiorespiratory and lower-body endurance"},
        {"Activity": "Rowing Machine", "Duration": 30.0, "Heart_Rate": 145.0, "Body_Temp": 39.4, "Focus": "Full-body muscular and cardiovascular power"},
        {"Activity": "Freestyle Swimming", "Duration": 30.0, "Heart_Rate": 130.0, "Body_Temp": 38.8, "Focus": "Zero-impact aerobic capacity"},
        {"Activity": "Jump Rope", "Duration": 20.0, "Heart_Rate": 160.0, "Body_Temp": 39.7, "Focus": "Cardio speed and neuromuscular agility"},
        {"Activity": "Incline Treadmill Walk", "Duration": 40.0, "Heart_Rate": 115.0, "Body_Temp": 38.5, "Focus": "Zone 2 fat oxidation without joint strain"},
        {"Activity": "Power Yoga / Calisthenics", "Duration": 45.0, "Heart_Rate": 95.0, "Body_Temp": 37.8, "Focus": "Core stability, flexibility and active recovery"}
    ]

    rank_rows = []
    for ex in exercise_catalog:
        row_feat = {
            "Gender": gender_val,
            "Age": int(age),
            "Height": float(height),
            "Weight": float(weight),
            "Duration": ex["Duration"],
            "Heart_Rate": ex["Heart_Rate"],
            "Body_Temp": ex["Body_Temp"]
        }
        pred_c = max(0.0, float(model.predict(pd.DataFrame([row_feat]))[0]))
        b_rate = round(pred_c / ex["Duration"], 2)
        rank_rows.append({
            "Exercise Modality": ex["Activity"],
            "Duration (min)": ex["Duration"],
            "Avg Heart Rate (BPM)": ex["Heart_Rate"],
            "Predicted Energy (kcal)": round(pred_c, 1),
            "Burn Rate (kcal/min)": b_rate,
            "Primary Metabolic Focus": ex["Focus"]
        })

    df_ranked = pd.DataFrame(rank_rows).sort_values(by="Burn Rate (kcal/min)", ascending=False).reset_index(drop=True)
    df_ranked.index += 1
    df_ranked.index.name = "Rank"

    st.dataframe(df_ranked, use_container_width=True)

    # Visualization of Burn Rate by Activity
    fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=df_ranked.reset_index(),
        x="Burn Rate (kcal/min)",
        y="Exercise Modality",
        palette="Reds_r",
        ax=ax_bar
    )
    ax_bar.set_title(f"Predicted Calorie Burn Rate by Exercise (Age: {age}, Weight: {weight}kg)", fontsize=11, fontweight="bold")
    ax_bar.set_xlabel("Burn Rate (kcal / minute)", fontsize=10)
    ax_bar.set_ylabel("")
    plt.tight_layout()
    st.pyplot(fig_bar)


# ===========================================================================
# TAB 3: DIET & NUTRITION PLAN
# ===========================================================================
with tab3:
    st.subheader("Targeted Diet & Nutritional Planning")
    st.caption("Evidence-based metabolic calculations using the Mifflin-St Jeor clinical equation:")

    # BMR Calculation (Mifflin-St Jeor)
    if gender == "Male":
        bmr = (10.0 * weight) + (6.25 * height) - (5.0 * age) + 5.0
    else:
        bmr = (10.0 * weight) + (6.25 * height) - (5.0 * age) - 161.0
    bmr = round(bmr, 0)

    # TDEE Calculation (Baseline 1.35 factor + session expenditure)
    tdee = round((bmr * 1.35) + predicted_calories, 0)

    # Caloric Target based on Goal
    if "Fat Loss" in fitness_goal:
        target_calories = max(1400.0, round(tdee - 450.0, 0))
        target_desc = "Caloric Deficit Target (-450 kcal/day for sustainable fat loss of ~0.45 kg/week)"
        carb_pct, fat_pct = 0.40, 0.30
    elif "Hypertrophy" in fitness_goal:
        target_calories = round(tdee + 350.0, 0)
        target_desc = "Caloric Surplus Target (+350 kcal/day to support muscle hypertrophy and tissue repair)"
        carb_pct, fat_pct = 0.50, 0.25
    else:
        target_calories = tdee
        target_desc = "Caloric Maintenance Target (Equilibrium with TDEE to sustain current weight and conditioning)"
        carb_pct, fat_pct = 0.45, 0.30

    # Macronutrient Breakdown
    # Protein: 2.0g per kg bodyweight
    protein_grams = round(weight * 2.0, 0)
    protein_cals = protein_grams * 4.0

    fat_cals = round(target_calories * fat_pct, 0)
    fat_grams = round(fat_cals / 9.0, 0)

    carb_cals = max(0.0, target_calories - protein_cals - fat_cals)
    carb_grams = round(carb_cals / 4.0, 0)

    # Top Nutrition Metrics
    col_n1, col_n2, col_n3, col_n4 = st.columns(4)
    with col_n1:
        st.metric("Basal Metabolic Rate (BMR)", f"{int(bmr)} kcal/day")
    with col_n2:
        st.metric("Total Daily Energy (TDEE)", f"{int(tdee)} kcal/day")
    with col_n3:
        st.metric("Daily Calorie Target", f"{int(target_calories)} kcal/day")
    with col_n4:
        st.metric("Daily Protein Target", f"{int(protein_grams)} g/day")

    st.markdown(f"**Selected Strategy:** {target_desc}")

    # Macronutrient Allocation Table
    col_macro_tab, col_guidance = st.columns([1.2, 1])

    with col_macro_tab:
        st.markdown("#### Daily Macronutrient Allocation")
        macro_df = pd.DataFrame([
            {"Nutrient": "Protein", "Target (g)": f"{int(protein_grams)} g", "Calories": f"{int(protein_cals)} kcal", "Ratio": f"{round(protein_cals / target_calories * 100, 1)}%", "Role": "Muscle protein synthesis, cellular tissue repair, satiety"},
            {"Nutrient": "Carbohydrates", "Target (g)": f"{int(carb_grams)} g", "Calories": f"{int(carb_cals)} kcal", "Ratio": f"{round(carb_cals / target_calories * 100, 1)}%", "Role": "Glycogen restoration, aerobic energy, central nervous system fuel"},
            {"Nutrient": "Healthy Fats", "Target (g)": f"{int(fat_grams)} g", "Calories": f"{int(fat_cals)} kcal", "Ratio": f"{round(fat_cals / target_calories * 100, 1)}%", "Role": "Endocrine hormone production, cell membrane integrity, vitamin uptake"}
        ])
        st.dataframe(macro_df, hide_index=True, use_container_width=True)

    with col_guidance:
        st.markdown("#### Structured Meal Timing")
        st.markdown("""
        - **Pre-Workout (45-60 min prior):** 30-40g complex carbohydrates (oatmeal, banana, whole grain toast) + 300-500 mL water.
        - **Post-Workout (within 45 min):** 25-35g fast-digesting protein (whey/egg whites/tofu) paired with 40-50g simple carbs to replenish muscle glycogen.
        - **Hydration Recommendation:** Minimum 3.5 Liters daily, plus 500 mL per 30 minutes of elevated physical exertion.
        """)

    st.markdown("#### Wholesome Food Recommendations")
    st.markdown("""
    - **Quality Protein:** Chicken breast, turkey, eggs, wild salmon, paneer, tofu, lentils, Greek yogurt.
    - **Complex Carbohydrates:** Rolled oats, brown rice, sweet potatoes, quinoa, whole-wheat pita, berries.
    - **Essential Fats:** Extra virgin olive oil, avocado, raw almonds, walnuts, chia seeds.
    """)


# ===========================================================================
# TAB 4: LIFESTYLE RISK SCREENING & PREVENTIVE GUIDELINES
# ===========================================================================
with tab4:
    st.subheader("Lifestyle & Cardiovascular Risk Screening")
    st.caption("Automated clinical risk stratification based on recorded physiological parameters:")

    # 1. Cardiovascular Strain Evaluation
    if hr_pct >= 88.0 or heart_rate >= 175.0:
        cv_status = "Elevated Risk - High Cardiovascular Strain"
        cv_class = "risk-card-danger"
        cv_condition = "Risk of acute myocardial overexertion, excessive hemodynamic load, or unmonitored tachycardia."
        cv_advice = "Scale back workout intensity into Zone 2-3 (65-80% HR Max). Implement a 10-minute progressive warmup and cooldown. If dizzy or breathless, pause immediately."
    elif hr_pct >= 80.0:
        cv_status = "Moderate Caution - Anaerobic Threshold Strain"
        cv_class = "risk-card-warning"
        cv_condition = "High lactate accumulation and substantial cardiac demand; safe for conditioned individuals in interval format."
        cv_advice = "Ensure rest intervals between high-intensity sets. Limit continuous Zone 4 training to 20-30 minutes."
    else:
        cv_status = "Optimal - Healthy Cardiovascular Response"
        cv_class = "risk-card-normal"
        cv_condition = "Heart rate remained within safe physiological aerobic conditioning parameters."
        cv_advice = "Maintain current aerobic pacing to foster long-term cardiovascular health and capillary density."

    # 2. Exertional Heat Strain Evaluation
    if body_temp >= 40.5:
        heat_status = "High Risk - Exertional Hyperthermia"
        heat_class = "risk-card-danger"
        heat_condition = "Severe heat strain risk, potential heat exhaustion or exertional heat illness."
        heat_advice = "Cease physical exertion immediately. Hydrate with electrolyte-enriched fluids, rest in a cooled environment, and apply passive cooling."
    elif body_temp >= 39.8 and duration >= 40.0:
        heat_status = "Moderate Caution - Elevated Thermal Accumulation"
        heat_class = "risk-card-warning"
        heat_condition = "Prolonged core temperature elevation with sweat fluid and electrolyte loss."
        heat_advice = "Consume 250 mL of water or electrolyte solution every 15-20 minutes. Avoid exercising in unventilated or humid environments."
    else:
        heat_status = "Optimal - Regulated Thermal State"
        heat_class = "risk-card-normal"
        heat_condition = "Core temperature reflects normal physiological metabolic thermogenesis."
        heat_advice = "Continue standard hydration practices (500 mL pre-workout and post-workout fluid restoration)."

    # 3. Metabolic & Weight Risk Evaluation
    if bmi >= 30.0:
        bmi_status = "Elevated Risk - Class I/II Obesity Indicators"
        bmi_class = "risk-card-danger"
        bmi_condition = "Increased predisposition to Type 2 Diabetes, essential hypertension, dyslipidemia, and knee joint osteoarthritis."
        bmi_advice = "Favor non-weight-bearing cardiovascular exercise (indoor cycling, swimming, rowing) to preserve joint cartilage. Pair with a monitored 500 kcal daily deficit."
    elif bmi >= 25.0:
        bmi_status = "Moderate Risk - Overweight Metabolic Profile"
        bmi_class = "risk-card-warning"
        bmi_condition = "Moderate predisposition to visceral adiposity, reduced insulin sensitivity, and mild hypertension."
        bmi_advice = "Aim for 150-250 minutes of weekly aerobic exercise combined with resistance training and glycemic management."
    elif bmi < 18.5:
        bmi_status = "Caution - Underweight / Lean Tissue Deficiency"
        bmi_class = "risk-card-warning"
        bmi_condition = "Risk of relative energy deficiency, diminished muscle mass, lowered bone mineral density, and immune suppression."
        bmi_advice = "Prioritize progressive resistance training for hypertrophy; avoid excessive long-distance cardio; adopt a 300-500 kcal daily caloric surplus."
    else:
        bmi_status = "Optimal - Healthy Weight Category"
        bmi_class = "risk-card-normal"
        bmi_condition = "Associated with low chronic metabolic disease incidence and optimal physiological resilience."
        bmi_advice = "Sustain consistent physical conditioning and balanced whole-food nutrition."

    # 4. Overtraining & Recovery Deficit
    if duration >= 60.0 and heart_rate >= 150.0:
        over_status = "Elevated Risk - High Overtraining Accumulation"
        over_class = "risk-card-danger"
        over_condition = "Elevated systemic cortisol, muscle protein catabolism, central nervous system fatigue, and blunted recovery."
        over_advice = "Cap high-intensity cardiovascular sessions at 45 minutes. Guarantee 48 hours of recovery between strenuous sessions and sleep 7-9 hours nightly."
    elif duration >= 45.0 and heart_rate >= 140.0:
        over_status = "Moderate Caution - Demanding Training Volume"
        over_class = "risk-card-warning"
        over_condition = "Substantial glycogen depletion and delayed muscle soreness expected."
        over_advice = "Incorporate structured post-workout carbohydrate replenishment and active mobility recovery."
    else:
        over_status = "Optimal - Sustainable Session Volume"
        over_class = "risk-card-normal"
        over_condition = "Workout duration and strain align with positive physiological adaptation."
        over_advice = "Standard recovery protocol (24-48 hours before retraining the same muscle groups)."

    # Render 4 Clinical Cards
    st.markdown(f"""
    <div class="{cv_class}">
        <h4>1. Cardiovascular Load: {cv_status}</h4>
        <p><strong>Linked Condition:</strong> {cv_condition}</p>
        <p><strong>Prevention Advice:</strong> {cv_advice}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="{heat_class}">
        <h4>2. Thermal & Heat Strain: {heat_status}</h4>
        <p><strong>Linked Condition:</strong> {heat_condition}</p>
        <p><strong>Prevention Advice:</strong> {heat_advice}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="{bmi_class}">
        <h4>3. Metabolic Profile (BMI): {bmi_status}</h4>
        <p><strong>Linked Condition:</strong> {bmi_condition}</p>
        <p><strong>Prevention Advice:</strong> {bmi_advice}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="{over_class}">
        <h4>4. Volume & Recovery Load: {over_status}</h4>
        <p><strong>Linked Condition:</strong> {over_condition}</p>
        <p><strong>Prevention Advice:</strong> {over_advice}</p>
    </div>
    """, unsafe_allow_html=True)


# Footer
st.markdown("---")
st.markdown("<div style='text-align:center; color:#888; font-size:0.85rem;'>Calories Burnt Prediction & Health Intelligence System | Powered by Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn & Streamlit</div>", unsafe_allow_html=True)
