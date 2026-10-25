# RAMDEOBABA UNIVERSITY
### Department of Computer Science & Engineering

## PROJECT SYNOPSIS
**Session:** 2025–2026 | **Semester:** VI | **Programme:** B.Tech Computer Science & Engineering

---

### Project Title
# CALORIE BURNT PREDICTION & HEALTH INTELLIGENCE SYSTEM
*An End-to-End Supervised ML Pipeline, Multi-Exercise Ranking & Clinical Diet Stratification*

---

**Submitted by:**  
**Gaurang Sonone**  
*Roll No.: [Roll No. Placeholder]*  
Department of Computer Science & Engineering  
Ramdeobaba University, Nagpur  

**Under the Guidance of:**  
**[Guide Name, Designation]**  
Assistant Professor, Department of Computer Science & Engineering  
Ramdeobaba University, Nagpur  

**Date of Submission:** 25 / 10 / 2026

---

### Formatting Guidelines Specification

| Parameter | Specification |
| :--- | :--- |
| **Font** | Times New Roman |
| **Title** | 16 pt, Bold, Uppercase |
| **Section headings** | 14 pt, Bold |
| **Body text** | 12 pt, Regular |
| **References** | 10 pt, IEEE (numbered) |
| **Line spacing** | 1.5 |
| **Margins** | 1 inch, all sides |
| **Citation style** | IEEE (numbered) |
| **Maximum length** | 4 pages (this page + 3) |

<!-- PAGE BREAK -->

---
*Ramdeobaba University — Project Synopsis* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *Department of Computer Science & Engineering*
*(Page 2 of 4)*
---

### 1. Background and Motivation
Physical inactivity is a leading driver of chronic lifestyle disorders globally, including cardiovascular ailments, obesity, and metabolic syndrome. While consumer smartwatches have popularized daily activity monitoring, clinical studies indicate commercial trackers exhibit high error rates (27% to 93%) when quantifying caloric expenditure. These inaccuracies stem from reliance on static Metabolic Equivalent of Task (MET) lookup tables or basic linear formulas that fail to model non-linear physiological interactions. During prolonged physical exertion, cardiovascular drift and thermal heat accumulation cause metabolic expenditure to accelerate non-linearly with exercise duration and cardiac workload. Furthermore, individual anthropometric variations (e.g., body mass index and height) act as non-linear kinetic multipliers on movement cost. Supervised Machine Learning provides a robust computational paradigm capable of decoding high-dimensional non-linear interactions across multimodal biometric telemetry, turning crude calorie tracking into actionable physiological intelligence.

### 2. Problem Statement
Existing fitness trackers act primarily as passive, retrospective logging utilities that record past workouts without proactive comparative guidance or actionable health decision support. Furthermore, current platforms remain isolated from clinical nutrition and lifestyle risk screening. This project formulates exercise energy expenditure as an end-to-end Machine Learning problem: given a 7-dimensional biometric telemetry vector (Gender, Age, Height, Weight, Duration, Heart Rate, Body Temperature), predict continuous caloric burn using trained tree ensembles, dynamically rank multi-exercise modalities, and synthesize personalized dietary and clinical safety recommendations.

### 3. Objectives
* **Data Hygiene & Leak-Free Engineering:** Acquire and merge 15,000 telemetry instances with zero data loss, enforcing an 80/20 train-test partition strictly prior to feature standardization to prevent data leakage.
* **Supervised Algorithmic Benchmarking:** Train and rigorously benchmark four syllabus regressors (Linear Regression, Decision Tree, Random Forest, and KNN) via 5-Fold Cross-Validation, achieving $R^2 \ge 0.99$.
* **Advanced Unsupervised & Concept Learning:** Uncover latent metabolic clusters using 2D PCA and K-Means ($K=3$), and derive minimal high-burn physiological conditions using Find-S inductive concept learning.
* **Multi-Exercise ML Recommender:** Predict calories and burn velocity (kcal/min) across 8 exercise modalities for the user's specific biometric profile.
* **Closed-Loop Clinical Nutrition & Safety Screening:** Synthesize exact daily protein and carbohydrate targets via Mifflin-St Jeor TDEE, and provide point-to-point cardiovascular and thermal risk flags.
* **Interactive Cloud Deployment:** Deploy an open-access, sub-second inference Streamlit web application.

### 4. Literature Review and Research Gap
Wearable energy expenditure estimation has transitioned from basic accelerometry to multimodal telemetry fusion:

| Sr. | Author(s) & Year | Title / Focus | Methodology / Key Contribution | Limitation / Gap |
| :---: | :--- | :--- | :--- | :--- |
| 1 | Staudenmayer et al. (2020) [1] | Wearable sensor energy expenditure modeling | Evaluated linear models and random forests on accelerometry | Omitted core thermal telemetry; high variance in high-intensity intervals. |
| 2 | Altini & Amft (2021) [2] | Wearable energy estimation comparative study | Compared gradient boosting and deep learning on telemetry | High model complexity; lack of integrated nutrition or proactive planning. |
| 3 | Zhang et al. (2022) [3] | Multimodal sensor fusion during intense activity | Fused optical PPG heart rate and skin temperature telemetry | Purely retrospective logging; no multi-exercise ranking decision support. |
| 4 | Patel & Gupta (2023) [4] | Data leakage prevention in biomedical ML | Analyzed methodological pipeline flaws in health data science | Theoretical analysis without an end-to-end deployed software artifact. |

**Research Gap:** Prior literature focuses exclusively on retrospective regression or sensor analysis. There is a lack of an integrated framework uniting high-accuracy non-linear ML regression with proactive multi-exercise ranking, closed-loop macronutrient diet planning, and clinical risk stratification.

<!-- PAGE BREAK -->

---
*Ramdeobaba University — Project Synopsis* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *Department of Computer Science & Engineering*
*(Page 3 of 4)*
---

### 5. Proposed Methodology / Plan of Work
The system employs a modular, leak-free machine learning architecture:
* **Dataset Ingestion & Hygiene:** Inner-joining `exercise.csv` and `calories.csv` on `User_ID` ($N = 15,000$ zero-loss records). Variance Inflation Factor (VIF) diagnostics evaluate multicollinearity across physiological features.
* **Zero-Leakage Splitting:** An 80/20 train-test split is performed prior to feature standardization. The StandardScaler is fit exclusively on $X_{\text{train}}$ and applied to transform $X_{\text{test}}$.
* **Model Training & CV:** Four supervised models—Linear Regression (OLS), Decision Tree, Random Forest (100 trees), and KNN ($k=11$)—are evaluated via 5-Fold Cross-Validation, residual analytics, and test metrics ($R^2$, MAE, RMSE).
* **Unsupervised & Concept Learning:** 2D PCA (96.3% variance explained) and K-Means ($K=3$) discover physiological exertion clusters. The Find-S algorithm inductively bounds requirements for burning $\ge 150\text{ kcal}$.
* **Health Intelligence Engine:** Deploys the best model (Random Forest, $R^2 = 0.9980$) to drive multi-exercise ranking, Mifflin-St Jeor TDEE diet synthesis (exact protein/carbs), and clinical risk alerts.

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
│ 4 Syllabus Regressors Benchmark (5-Fold CV) ──► Best: Random Forest     │
│ (Linear Regression: 0.9673 | Decision Tree: 0.9928 | RF: 0.9980 | KNN) │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Deployed Streamlit Cloud Application (Joblib Serialized Pipeline)      │
│ • ML Exercise Ranking • TDEE Diet Engine (Protein/Carbs) • Risk Flags  │
└────────────────────────────────────────────────────────────────────────┘
```

### 6. Technology, Tools and Platforms
* **Languages & Frameworks:** Python 3.14, Scikit-Learn (models, cross-validation, PCA, metrics), Pandas and NumPy (data structures), Joblib (model persistence).
* **Visualization & UI:** Matplotlib and Seaborn (diagnostic plots), Streamlit (web application deployment).
* **Environment & Cloud:** VS Code / Antigravity IDE, Git, GitHub repository, and Streamlit Cloud.
* **Benchmark Dataset:** Verified 15,000-instance Exercise Telemetry dataset (`exercise.csv` and `calories.csv`).

### 7. Expected Outcomes, Deliverables and Functional Specifications
* **Outcomes:** A validated Random Forest model achieving $R^2 = 0.9980$ and $\text{MAE} = 1.7\text{ kcal}$, outperforming linear baselines.
* **Software Deliverables:** (i) A production Streamlit web application providing real-time inference, (ii) an automated ML exercise ranking table evaluating 8 modalities, (iii) a clinical diet engine calculating Mifflin-St Jeor TDEE with exact daily protein ($2.0\text{ g/kg}$) and carbohydrate requirements, (iv) point-to-point cardiovascular and thermal safety alert screening, and (v) a reproducible 30-cell Jupyter notebook.

### 8. Project Scope
* **Applicability:** Personalized fitness training, gym biometric telemetry monitoring, athletic workload optimization, and preventive metabolic health tracking.
* **Boundaries & Constraints:** Designed for acute steady-state aerobic and conditioning sessions up to 60 minutes based on wearable telemetry; does not process raw continuous millisecond ECG waveforms or diagnose chronic cardiovascular pathology.

<!-- PAGE BREAK -->

---
*Ramdeobaba University — Project Synopsis* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *Department of Computer Science & Engineering*
*(Page 4 of 4)*
---

### 9. Project Timeline
The project is executed over an eight-week engineering schedule covering data hygiene, ML benchmarking, and deployment:

| Phase | Duration | Milestone / Deliverable |
| :--- | :---: | :--- |
| Literature review & requirement analysis | Week 1 – 2 | Dataset acquisition ($N=15,000$), exploratory data analysis, research gap analysis |
| Design & methodology finalisation | Week 3 – 4 | Zero-loss join verification, VIF collinearity diagnostics, split-before-scaling setup |
| Implementation & model development | Week 5 – 6 | Training 4 regressors, 5-Fold Cross Validation, PCA/K-Means, Find-S implementation |
| Testing, evaluation & refinement | Week 7 | Benchmark leaderboard generation, residual analysis, clinical diet & risk validation |
| Documentation & final submission | Week 8 | Streamlit Cloud deployment, GitHub repository finalization, academic synopsis report |

### 10. References
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
| :--- | :--- |
| *[Roll No.]* | **Gaurang Sonone** |
| &nbsp; | &nbsp; |

---

**Approved by:**  
**(Name of Guide and Signature)**  
[Guide Name, Designation]  
Department of Computer Science & Engineering  
Ramdeobaba University, Nagpur  

*Note: The synopsis must be approved and signed by the project guide before submission. A signed hard copy is to be submitted to the Project Coordinator by the deadline notified for the session.*
