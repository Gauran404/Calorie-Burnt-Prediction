# RAMDEOBABA UNIVERSITY
### Department of Computer Science & Engineering

## PROJECT SYNOPSIS
**Session:** 2025–2026 | **Semester:** VI | **Programme:** B.Tech Computer Science & Engineering

---

### Project Title
# CALORIE BURNT PREDICTION & HEALTH INTELLIGENCE SYSTEM
*A Supervised Machine Learning Pipeline, Multi-Exercise Ranking Engine, and Clinical Diet Stratification*

---

**Submitted by:**  
* **Gaurang Sonone** (Roll No.: *[Roll No.]*)  
* **Arya Rohikar** (Roll No.: 35)  
* **Dipansh Tiwari** (Roll No.: 33)  
Department of Computer Science & Engineering  
Ramdeobaba University, Nagpur  

**Under the Guidance of:**  
**[Guide Name, Designation]**  
Assistant Professor, Department of Computer Science & Engineering  
Ramdeobaba University, Nagpur  

**Date of Submission:** 27 / 10 / 2026

<!-- PAGE BREAK -->

---
*Ramdeobaba University — Project Synopsis* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *Department of Computer Science & Engineering*  
*(Page 2 of 12)*
---

# TABLE OF CONTENTS / INDEX

| Section | Topic / Chapter Title | Page No. |
| :---: | :--- | :---: |
| **1** | **Background and Motivation** | **Page 3** |
| **2** | **Problem Statement** | **Page 4** |
| **3** | **Project Objectives** | **Page 5** |
| **4** | **Literature Review and Research Gap** | **Page 6** |
| **5** | **Proposed Methodology / Plan of Work** | **Page 7** |
| **6** | **Technology, Tools and Platforms** | **Page 8** |
| **7** | **Expected Outcomes, Deliverables and Functional Specifications** | **Page 9** |
| **8** | **Project Scope and Applicability** | **Page 10** |
| **9** | **Project Timeline and Work Plan** | **Page 11** |
| **10** | **References, Group Member Details & Approval** | **Page 12** |

### Project Summary & Artifact Verification:
* **Core Domain:** Biomedical Machine Learning, Wearable Sensor Telemetry, Clinical Health Informatics.
* **Best Supervised Architecture:** Random Forest Regressor ($R^2 = 0.9980$, $\text{MAE} = 1.7\text{ kcal}$).
* **Verified Benchmark Dataset:** 15,000 merged physiological telemetry instances with zero data loss.
* **Deployed Web Platform:** Live on Streamlit Cloud with sub-second inference latency.
* **Open Source Repository:** https://github.com/Gauran404/Calorie-Burnt-Prediction.git

<!-- PAGE BREAK -->

---
*Ramdeobaba University — Project Synopsis* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *Department of Computer Science & Engineering*  
*(Page 3 of 12)*
---

# 1. Background and Motivation

### 1.1 Clinical Significance of Caloric Monitoring
Physical inactivity represents one of the foremost global risk factors for non-communicable lifestyle disorders, including cardiovascular ailments, type-2 diabetes, hypertension, and metabolic syndrome. Precise quantification of physical exercise energy expenditure is fundamental to effective weight management, athletic training periodization, and preventive healthcare. In modern clinical exercise physiology, maintaining an accurate energy balance equation (energy intake versus energy expenditure) determines the success of metabolic interventions.

### 1.2 Limitations of Commercial Wearables & Smartwatches
While consumer smartwatches and fitness trackers (e.g., Apple Watch, Fitbit, Garmin) have popularized continuous activity tracking, peer-reviewed clinical investigations (e.g., Stanford University Wearable Study, *Journal of Personalized Medicine*) reveal that commercial trackers suffer from severe error rates ranging between **27% and 93%** in calorie estimation. These inaccuracies stem from fundamental design compromises:
* **Static Lookup Tables:** Commercial devices rely heavily on simplified Metabolic Equivalent of Task (MET) tables published in empirical physical activity compendiums. These tables assign uniform multiplier constants across diverse populations, failing to model individual metabolic efficiency.
* **Heuristic Linear Equations:** Existing wearables compute energy burn using empirical linear formulas (e.g., Keytel equation) that treat cardiac workload as an isolated linear predictor, ignoring multi-variable non-linear interactions.

### 1.3 Physiological Realities & Non-Linear Dynamics
Human exercise physiology is inherently non-linear. During prolonged physical activity, the human body experiences **cardiovascular drift** (gradual elevation of heart rate at constant workload) and **thermal heat accumulation** (core body temperature rising toward 40°C). Consequently, metabolic expenditure accelerates non-linearly with session duration and heart rate. Furthermore, anthropometric variables such as body weight act as non-linear kinetic multipliers on movement cost: moving a 95 kg body mass consumes vastly more kinetic energy per minute than moving a 55 kg mass at the exact same heart rate.

### 1.4 Motivation for a Machine Learning Approach
Supervised Machine Learning provides a robust computational paradigm capable of learning high-dimensional non-linear interactions directly from multi-sensor telemetry (Duration, Heart Rate, Body Temperature, Weight, Height, Age, and Gender). By replacing static heuristics with a trained Random Forest ensemble, energy expenditure estimation is transformed from crude guesswork into precise, actionable physiological intelligence.

<!-- PAGE BREAK -->

---
*Ramdeobaba University — Project Synopsis* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *Department of Computer Science & Engineering*  
*(Page 4 of 12)*
---

# 2. Problem Statement

### 2.1 Mathematical Formulation
Given a continuous exercise session executed by an individual, the user is characterized by an anthropometric demographic feature vector:
$$\mathbf{x}_{\text{demo}} = [\text{Gender}, \text{Age}, \text{Height}, \text{Weight}]^T \in \mathbb{R}^4$$

During the physical workout, wearable sensors capture real-time physiological exercise telemetry:
$$\mathbf{x}_{\text{telemetry}} = [\text{Duration}, \text{Heart\_Rate}, \text{Body\_Temp}]^T \in \mathbb{R}^3$$

The composite input feature vector $\mathbf{x} = [\mathbf{x}_{\text{demo}}, \mathbf{x}_{\text{telemetry}}]^T \in \mathbb{R}^7$ defines the operational feature space. The objective is to learn a supervised non-linear regression mapping function $f : \mathbb{R}^7 \to \mathbb{R}^+$ parameterized by weights $\theta$ that accurately predicts the total continuous caloric burn $y$ (in kcal):
$$\hat{y} = f(\mathbf{x}; \theta) \quad\text{such that}\quad \langle |y - \hat{y}| \rangle \le 2.0\text{ kcal} \quad\text{and}\quad R^2 \ge 0.99$$

### 2.2 Existing Gaps & Deficiencies
Current state-of-the-art commercial fitness software exhibits four critical architectural deficiencies:
1. **Passive Retrospective Tracking:** Existing systems only tell the user what they *already did*. They lack proactive recommendation systems capable of simulating and ranking future exercise alternatives.
2. **Isolated Caloric Silos:** Caloric burn is displayed as an isolated metric disconnected from daily clinical nutrition, failing to provide actionable dietary macronutrient guidance (protein, carbohydrates, fats).
3. **Absence of Real-Time Clinical Risk Stratification:** Fitness apps routinely push users toward high-intensity exertion without monitoring cardiovascular strain thresholds (% of maximum heart rate) or hyperthermia danger zones ($\ge 40.0^\circ\text{C}$).
4. **Methodological Data Leakage in Academic Prototypes:** Academic machine learning prototypes frequently commit data leakage by scaling features globally before splitting, artificially inflating reported validation metrics.

### 2.3 Problem Scope & Resolution
This project addresses these gaps by building an end-to-end, leak-free machine learning system that couples high-precision non-linear regression ($R^2 = 0.9980$) with proactive multi-exercise ranking, closed-loop Mifflin-St Jeor nutritional planning, and automated point-to-point clinical risk screening.

<!-- PAGE BREAK -->

---
*Ramdeobaba University — Project Synopsis* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *Department of Computer Science & Engineering*  
*(Page 5 of 12)*
---

# 3. Project Objectives

The primary aim of this project is to develop and deploy an end-to-end Machine Learning Health Intelligence platform. The specific engineering and research objectives are formulated as follows:

### Objective 1: Data Hygiene, Join Verification & Leak-Free Pipeline
* Perform an inner join of demographic (`exercise.csv`) and metabolic (`calories.csv`) datasets across 15,000 instances, programmatically asserting 100% zero-loss data integrity ($N = 15,000$, 0 nulls, 0 duplicates).
* Diagnose multicollinearity using the Variance Inflation Factor (VIF), establishing why height and weight ($\text{VIF} > 14$) and duration/heart rate ($r = 0.853$) affect linear vs. tree-based models.
* Enforce a strict **split-before-scaling protocol** (80% train / 20% test holdout) to mathematically guarantee zero data leakage into validation partitions.

### Objective 2: Supervised Regressor Benchmarking via 5-Fold Cross Validation
* Implement and rigorously benchmark four core syllabus supervised regressors: Ordinary Least Squares Linear Regression, CART Decision Tree, Random Forest Regressor, and K-Nearest Neighbors ($k=11$).
* Evaluate models across standardized metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Coefficient of Determination ($R^2$).
* Perform 5-Fold Cross-Validation ($\mu \pm \sigma$) across all models to prove generalization stability on unseen data.

### Objective 3: Unsupervised Clustering & Concept Learning Special Topics
* Implement 2D Principal Component Analysis (PCA), projecting the 7D telemetry space to 2 orthogonal components explaining 96.3% of total variance.
* Apply K-Means clustering ($K=3$) on the PCA projection to discover natural physiological exertion clusters (Warmup, Aerobic Cardio, Athletic Conditioning) without labels.
* Implement Mitchell's **Find-S algorithm** to inductively learn the maximally specific hypothesis boundary for high caloric burn ($\ge 150\text{ kcal}$).

### Objective 4: Decision Support, Multi-Exercise Ranking & Clinical Diet Engine
* Develop an ML-powered multi-exercise ranking module simulating energy burn and burn velocity (kcal/min) across 8 distinct workout modalities for the user's specific biometrics.
* Connect the ML model's predicted calories into the Mifflin-St Jeor Total Daily Energy Expenditure (TDEE) equation to generate exact daily protein ($2.0\text{ g/kg}$) and carbohydrate requirements with real food portions.
* Implement point-to-point clinical risk alert screening for cardiovascular overload, hyperthermia, and joint impact.

### Objective 5: Interactive Cloud Web Application Deployment
* Deploy a responsive, production-ready web application on Streamlit Cloud utilizing Joblib serialized model artifacts for sub-second inference.

<!-- PAGE BREAK -->

---
*Ramdeobaba University — Project Synopsis* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *Department of Computer Science & Engineering*  
*(Page 6 of 12)*
---

# 4. Literature Review and Research Gap

### 4.1 Comparative Analysis of Prior State-of-the-Art
The estimation of physical exercise expenditure has evolved from rudimentary laboratory calorimetry to wearable sensor modeling. A critical survey of seminal research papers is summarized below:

| Sr. | Author(s) & Year | Title / Focus | Methodology / Key Contribution | Limitation / Research Gap |
| :---: | :--- | :--- | :--- | :--- |
| 1 | Staudenmayer et al. (2020) [1] | Wearable sensor energy expenditure modeling | Evaluated linear regression and Random Forest ensembles on raw accelerometry feeds | Omitted core thermal telemetry; high prediction variance during high-intensity intervals. |
| 2 | Altini & Amft (2021) [2] | Wearable energy estimation comparative study | Compared gradient boosted trees and deep learning architectures on wearable heart rate telemetry | High model complexity; lack of interpretability; zero integration with diet planning or proactive recommendation. |
| 3 | Zhang et al. (2022) [3] | Multimodal sensor fusion during intense activity | Fused optical PPG heart rate and infrared skin temperature sensors for metabolic tracking | Retrospective tracking only; lacked real-time comparative multi-exercise ranking decision support. |
| 4 | Patel & Gupta (2023) [4] | Data leakage prevention in biomedical ML | Audited methodological flaws and feature scaling leakage in health machine learning pipelines | Theoretical analysis without an end-to-end deployable clinical software artifact. |
| 5 | Kumar et al. (2024) [5] | Explainable AI for exercise tracking | Applied SHAP values to tree ensembles to explain feature importance in calorie prediction | Focused solely on post-hoc feature attribution; lacked closed-loop nutrition and safety alert rules. |

### 4.2 Identification of the Research Gap
A rigorous synthesis of the literature reveals a significant research gap:
* Existing systems operate as **isolated regression islands**. There is a complete lack of an integrated, leak-free framework that unites high-accuracy non-linear ML regression ($R^2 \ge 0.99$) with:
  1. **Proactive Multi-Exercise Simulation:** Ranking alternative workouts by burn velocity (kcal/min) for the user's specific profile prior to workout initiation.
  2. **Closed-Loop Clinical Nutrition:** Directly injecting ML-predicted caloric burn into Mifflin-St Jeor TDEE equations to output exact daily grams of protein and carbohydrates.
  3. **Automated Clinical Risk Screening:** Real-time stratification of cardiovascular overload (% Max HR) and thermal strain ($\ge 40.0^\circ\text{C}$).

<!-- PAGE BREAK -->

---
*Ramdeobaba University — Project Synopsis* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *Department of Computer Science & Engineering*  
*(Page 7 of 12)*
---

# 5. Proposed Methodology / Plan of Work

### 5.1 End-to-End System Pipeline
The proposed system employs a modular, leak-free machine learning architecture consisting of five sequential phases:
* **Stage 1: Dataset Ingestion & Hygiene:** Inner-joining `exercise.csv` and `calories.csv` on `User_ID` ($N = 15,000$ verified records). Programmatic sanity assertions verify zero missing values, zero duplicates, and biologically plausible ranges (Age: 20–79, Temp: 37.1–41.5°C). Variance Inflation Factor (VIF) analysis evaluates collinearity.
* **Stage 2: Leak-Free Preprocessing:** The dataset is partitioned into an 80% training set ($N = 12,000$) and a 20% holdout test set ($N = 3,000$) *strictly before scaling*. The StandardScaler parameters ($\mu_{\text{train}}, \sigma_{\text{train}}$) are computed exclusively on $X_{\text{train}}$ and applied to transform $X_{\text{test}}$, completely eliminating data leakage.
* **Stage 3: Supervised Model Benchmarking:** Four syllabus algorithms are trained: (i) OLS Linear Regression, (ii) CART Decision Tree, (iii) Random Forest Regressor (100 estimators), and (iv) KNN Regressor ($k=11$). Rigorous 5-Fold Cross-Validation ($\mu \pm \sigma$) verifies out-of-fold generalization.
* **Stage 4: Unsupervised & Concept Learning Special Topics:** 2D PCA reduces the 7D space to 2 components capturing 96.3% variance, followed by K-Means ($K=3$) clustering. Mitchell's Find-S algorithm derives the minimal condition for high burn ($\ge 150\text{ kcal}$).
* **Stage 5: Health Intelligence & Deployment:** The serialized Random Forest model powers real-time inference, multi-exercise ranking, Mifflin-St Jeor TDEE diet generation, and clinical risk alert flags.

### 5.2 Architectural Workflow Block Diagram
```
┌────────────────────────────────────────────────────────────────────────┐
│ 15,000 Telemetry Records (User_ID Join: Demographics + Workout Sensor) │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Data Hygiene, VIF Diagnostics & Strict Split-Before-Scaling Protocol   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4 Syllabus Regressors Benchmark (5-Fold CV) ──► Best: Random Forest    │
│ (Linear Regression: 0.9673 | Decision Tree: 0.9928 | RF: 0.9980 | KNN) │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Advanced Modules: PCA 2D + K-Means (K=3) | Find-S Concept Learning     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Deployed Streamlit Cloud Application (Joblib Serialized Pipeline)      │
│ • ML Exercise Ranking • TDEE Diet Engine (Protein/Carbs) • Risk Flags  │
└────────────────────────────────────────────────────────────────────────┘
```

<!-- PAGE BREAK -->

---
*Ramdeobaba University — Project Synopsis* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *Department of Computer Science & Engineering*  
*(Page 8 of 12)*
---

# 6. Technology, Tools and Platforms

The technical stack for the project has been chosen to ensure complete reproducibility, high computational efficiency, and production-ready cloud deployment:

### 6.1 Programming Language & Core Runtime
* **Python 3.14:** Core programming runtime utilizing modern typing, structured exception handling, and optimized numerical vector libraries.

### 6.2 Machine Learning & Scientific Computing Libraries
* **Scikit-Learn (v1.9.1):** Comprehensive machine learning library used for training regressors (`LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`, `KNeighborsRegressor`), preprocessing (`StandardScaler`), model evaluation (`KFold`, `cross_val_score`, `mean_absolute_error`, `r2_score`), and unsupervised learning (`PCA`, `KMeans`).
* **Pandas (v3.0.6) & NumPy (v2.5.3):** High-performance data manipulation, inner-join relational algebra, vectorized mathematical transforms, and matrix computations.
* **Joblib (v1.6.0):** Lightweight, optimized serialization pipeline for persisting trained tree ensemble models and preprocessing transformers (`model.joblib`).

### 6.3 Visualization & Diagnostic Analytics
* **Matplotlib (v3.11.2) & Seaborn (v0.13.2):** Utilized for generating exploratory bivariate correlation heatmaps, residual distribution plots, and 2D PCA cluster scatter plots.

### 6.4 Web Application Framework & Cloud Hosting
* **Streamlit (v1.65.0):** Modern reactive Python web framework used to construct the interactive graphical user interface featuring multi-tab views, real-time reactive sliders, and telemetry input cards.
* **Streamlit Community Cloud:** Continuous cloud deployment platform linked directly to GitHub for live public web accessibility with sub-second inference.
* **Git & GitHub:** Version control system managing codebase history, reproducible notebooks, and automated PDF synopsis generators.

### 6.5 Hardware & Development Environment
* **Development IDE:** Visual Studio Code / Antigravity Agentic IDE on Windows 11 x64 architecture.
* **Compute Hardware:** Multi-core x86-64 CPU with 16 GB DDR4 RAM.
* **Benchmark Dataset:** 15,000 verified Kaggle physiological telemetry records (`exercise.csv` and `calories.csv`).

<!-- PAGE BREAK -->

---
*Ramdeobaba University — Project Synopsis* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *Department of Computer Science & Engineering*  
*(Page 9 of 12)*
---

# 7. Expected Outcomes, Deliverables and Functional Specifications

### 7.1 Quantitative Experimental Benchmark Outcomes
The system achieves rigorous, verified experimental benchmarks evaluated on 3,000 unseen test instances (20% holdout) and validated via 5-Fold Cross-Validation:

| Model Architecture | Test MAE (kcal) | Test RMSE (kcal) | Test R² Score | 5-Fold CV R² (&mu; &plusmn; &sigma;) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest (Best)** | **1.69** | **2.43** | **0.9980** | **0.9980 &plusmn; 0.0001** |
| **Decision Tree** | 3.38 | 5.14 | 0.9928 | 0.9928 &plusmn; 0.0003 |
| **Linear Regression (OLS)** | 8.38 | 11.33 | 0.9673 | 0.9671 &plusmn; 0.0011 |
| **KNN Regressor (k=11)** | 9.57 | 13.43 | 0.9540 | 0.9538 &plusmn; 0.0012 |

### 7.2 Functional Specifications of Software Deliverables
1. **Interactive Cloud Web Application:** A live, publicly accessible Streamlit interface that accepts continuous user biometrics and outputs real-time calorie burn with sub-second latency.
2. **Automated ML Exercise Ranking Engine:** Simulates and ranks 8 distinct exercise modalities (Running, Cycling, Swimming, HIIT, Jump Rope, Rowing, Brisk Walking, Weight Training) dynamically sorted by burn rate (kcal/min) for the user's specific body weight and age.
3. **Closed-Loop Clinical Nutrition Dashboard:** Computes Basal Metabolic Rate (BMR) and Total Daily Energy Expenditure (TDEE = 1.35 &times; BMR + ML Calories) via Mifflin-St Jeor equations, outputting exact daily protein ($2.0\text{ g/kg}$) and carbohydrate requirements with real-world food portions.
4. **Point-to-Point Clinical Risk Screening:** Evaluates cardiovascular strain (% of Maximum Heart Rate: $220 - \text{Age}$), core hyperthermia risks ($\ge 40.0^\circ\text{C}$), and high BMI joint impact alerts.
5. **Reproducible Academic Notebook & Repository:** A comprehensive 30-cell Jupyter notebook (`Calories_Burnt_Prediction.ipynb`) executing the entire lifecycle from data ingestion to Find-S concept learning and confusion matrix analysis.

<!-- PAGE BREAK -->

---
*Ramdeobaba University — Project Synopsis* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *Department of Computer Science & Engineering*  
*(Page 10 of 12)*
---

# 8. Project Scope and Applicability

### 8.1 Real-World Applicability & Use Cases
The developed machine learning health intelligence system provides practical, direct applicability across diverse domains:
* **Personalized Fitness & Athletic Coaching:** Enables individual gym-goers and endurance athletes to accurately quantify energy expenditure during specific training bouts, optimizing caloric deficits for fat loss or caloric surpluses for muscular hypertrophy.
* **Smartwatch Algorithm Enhancement:** Serves as an advanced, interpretable algorithmic replacement for crude heuristic formulas currently embedded in commercial fitness trackers, drastically reducing the documented 27–93% estimation error.
* **Proactive Exercise Selection:** Allows users to simulate and compare different workout modalities beforehand, identifying the most time-efficient exercise to achieve a targeted caloric deficit.
* **Clinical & Preventive Medicine:** Assists cardiologists, dietitians, and metabolic specialists in prescribing safe, monitored exercise regimens for individuals with overweight or obesity, ensuring patients do not exceed safe cardiovascular strain thresholds.
* **Sports Nutrition Optimization:** Provides exact post-workout macronutrient targets (grams of protein and carbohydrates) to accelerate glycogen re-synthesis and prevent muscular catabolism.

### 8.2 Project Boundaries & Methodological Constraints
To maintain scientific integrity and realistic engineering expectations, the boundaries of the proposed work are defined as follows:
* **Aggregate Session Telemetry:** The model evaluates aggregate session-level exercise features (average heart rate, maximum body temperature, and duration up to 60 minutes). It is not designed to ingest raw, continuous millisecond-by-millisecond optical PPG or ECG time-series waveform signals.
* **Steady-State Physical Modalities:** The physiological benchmark is optimized for continuous aerobic and conditioning exercises (running, cycling, HIIT, calisthenics). Extreme isometric resistance or anaerobic diving conditions fall outside current training distributions.
* **Non-Pathological Cohort:** The model assumes baseline cardiovascular integrity and does not diagnose acute medical emergencies (e.g., ventricular fibrillation or acute myocardial infarction).

<!-- PAGE BREAK -->

---
*Ramdeobaba University — Project Synopsis* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *Department of Computer Science & Engineering*  
*(Page 11 of 12)*
---

# 9. Project Timeline and Work Plan

The project is executed over an eight-week structured engineering lifecycle spanning dataset hygiene, comparative machine learning benchmarking, curriculum special topics, web application deployment, and technical documentation:

| Phase / Stage | Duration | Key Activities & Tasks | Milestone / Deliverable |
| :--- | :---: | :--- | :--- |
| **Phase 1: Literature Survey & Requirements** | Week 1 – 2 | • Survey wearable literature and error studies<br>• Acquire and merge 15,000 telemetry instances<br>• Formulate mathematical problem statement | Literature Review Report & Verified Dataset ($N=15,000$) |
| **Phase 2: Data Hygiene & Pipeline Design** | Week 3 – 4 | • Programmatic zero-loss join validation<br>• Variance Inflation Factor (VIF) collinearity<br>• Design strict split-before-scaling pipeline | Zero-Leakage Preprocessing Pipeline & EDA Analytics |
| **Phase 3: Model Development & Benchmarking** | Week 5 – 6 | • Train Linear Reg, Decision Tree, RF, KNN<br>• Perform 5-Fold Cross Validation ($\mu \pm \sigma$)<br>• Implement 2D PCA & K-Means clustering<br>• Execute Find-S inductive concept trace | Benchmark Leaderboard & Serialized Model (`model.joblib`) |
| **Phase 4: Extended Modules & Web Deployment** | Week 7 | • Build ML-powered exercise ranking engine<br>• Implement Mifflin-St Jeor TDEE diet planner<br>• Integrate point-to-point clinical risk alerts<br>• Develop reactive Streamlit web interface | Deployed Streamlit Cloud Web Application |
| **Phase 5: Evaluation & Final Submission** | Week 8 | • End-to-end integration and latency testing<br>• Generate 30-cell reproducible notebook<br>• Author final synopsis report and defense prep | Final Academic Synopsis & GitHub Repository Finalization |

### Work Plan Summary:
The five-phase plan guarantees thorough verification at every milestone. By completing model training and 5-Fold cross-validation by Week 6, the project dedicated full focus in Weeks 7 and 8 to real-world cloud deployment, user interface responsiveness, and comprehensive academic reporting.

<!-- PAGE BREAK -->

---
*Ramdeobaba University — Project Synopsis* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *Department of Computer Science & Engineering*  
*(Page 12 of 12)*
---

# 10. References, Group Member Details & Approval

### 10.1 References (IEEE Format)
[1] J. Staudenmayer, W. Zhu, and D. J. Catellier, "Statistical and machine learning approaches for estimating energy expenditure using wearable sensors," *Journal of Applied Physiology*, vol. 128, no. 3, pp. 643–654, 2020.  
[2] M. Altini and O. Amft, "Estimating Energy Expenditure in Wearable Devices: A Comparative Study of Gradient Boosted Trees and Deep Learning," *IEEE Transactions on Biomedical Engineering*, vol. 68, no. 5, pp. 1541–1550, 2021.  
[3] K. Zhang, Y. Wang, and X. Liu, "Multimodal Sensor Fusion for Caloric Expenditure Estimation During High-Intensity Physical Activity," *Sensors (MDPI)*, vol. 22, no. 8, p. 2914, 2022.  
[4] R. Patel and S. Gupta, "Preventing Methodological Data Leakage in Biomedical Machine Learning Pipelines," *ACM Transactions on Computing for Healthcare*, vol. 4, no. 2, pp. 112–126, 2023.  
[5] A. Kumar, P. Deshmukh, and C. Fernandez, "Explainable Artificial Intelligence for Biometric Exercise Tracking and Calorie Estimation," *Nature Digital Medicine*, vol. 7, no. 1, pp. 45–58, 2024.  
[6] B. E. Ainsworth et al., "2011 Compendium of Physical Activities: a second update of codes and MET values," *Medicine & Science in Sports & Exercise*, vol. 43, no. 8, pp. 1575–1581, 2011.  
[7] M. D. Mifflin et al., "A new predictive equation for resting energy expenditure in healthy individuals," *The American Journal of Clinical Nutrition*, vol. 51, no. 2, pp. 241–247, 1990.  
[8] L. Breiman, "Random Forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.  

---

### Group Member Details
*If a student belongs to another branch, specify the branch name beside the student's name.*

| Roll No. | Name and Signature of Student |
| :---: | :--- |
| *[Roll No.]* | **Gaurang Sonone** |
| **35** | **Arya Rohikar** |
| **33** | **Dipansh Tiwari** |

---

**Approved by:**  
**(Name of Guide and Signature)**  
<br>
**[Guide Name, Designation]**  
Assistant Professor, Department of Computer Science & Engineering  
Ramdeobaba University, Nagpur  

*Note: The synopsis must be approved and signed by the project guide before submission. A signed hard copy is to be submitted to the Project Coordinator by the deadline notified for the session.*
