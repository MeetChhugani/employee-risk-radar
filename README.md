# 👥 Employee Risk Radar

An AI-powered HR analytics platform designed to predict employee attrition risks, explain model decisions using Explainable AI (SHAP), and generate tailored retention strategies using a Conversational LLM Advisor.

---

## 📊 Key Highlights & Metrics

*   **Ensemble Classifier Architecture**: Soft-voting ensemble combining hyperparameter-tuned **XGBoost**, **LightGBM**, and **CatBoost** classifiers optimized via **Optuna**.
*   **Imbalance Correction & Optimization**: Resolves class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique) and optimizes decision boundaries at a threshold of **`0.25`** (maximizing recall for actual attrition cases).
*   **Boosted Classification Scores**:
    *   **Test Set Accuracy**: **`84.35%`** (boosted from the baseline 82.0%)
    *   **Attrition Recall**: **`49.0%`** (increased by **+17.0%** absolute over the baseline)
    *   **Attrition F1-score**: **`0.50`** (boosted from the baseline 0.36)
    *   **5-Fold CV ROC-AUC**: **`0.79558`**
*   **Explainable AI (SHAP)**: Fully integrated local and global Shapley values directly in the application to translate black-box model decisions into clear feature-level contributions.
*   **Generative AI Retention Advisor**: Integrated **Groq (Llama 3.3)** to automatically analyze risk factor contributions and provide personalized weekly retention strategies and protocols via an interactive chatbot interface.

---

## 🛠️ Project Features

### 1. Risk Forecasting Dashboard
*   **Demographic & Satisfaction Analysis**: Custom multi-tab input panels for employee age, marital status, monthly income, job satisfaction, work-life balance, and overtime status.
*   **Preset Profiles**: Includes interactive templates to simulate common HR scenarios:
    *   🔴 *High-Risk Software Engineer*
    *   🟡 *Overworked Sales Representative*
    *   🟢 *Stable Executive Manager*

### 2. Explainable AI & Feature Impact
*   Renders local SHAP bar charts dynamically for the analyzed employee.
*   Identifies exactly which variables (e.g., Overtime, Low Income, Distance from Home) are driving the departure probability upward or downward.

### 3. HR Retention Chatbot
*   Generates instant actionable weekly targets and priority goals based on the SHAP explanation.
*   Accepts free-form conversational queries to help HR professionals research specific retention strategies.

---

## 📈 Model Performance & SHAP Visualizations

The underlying machine learning model is backed by extensive exploratory analysis and explainability benchmarks:

### Global Feature Importance (SHAP Bar Plot)
The most significant factors influencing attrition across the entire organization:
![SHAP Global Importance](shap_importance.png)

### Directional Summary Plot
Visualizes how high or low values of variables (like high overtime or low monthly income) impact departure risk:
![SHAP Summary Plot](shap_summary.png)

---

## 📂 Project Structure

```
├── .streamlit/             # Streamlit configuration
├── models/
│   └── attrition_model.pkl # Serialized XGBoost model, features, and optimal threshold
├── WA_Fn-UseC_-HR-Employee-Attrition.csv # IBM Watson HR Dataset
├── app.py                  # Main Streamlit web application
├── employee_attrition_predictor.ipynb  # EDA, SMOTE, Model Training, and SHAP calculations
├── requirements.txt        # Python package dependencies
└── README.md               # Project documentation
```

---

## ⚙️ Setup & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MeetChhugani/employee-risk-radar.git
   cd employee-risk-radar
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables. Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. Run the Streamlit web application:
   ```bash
   streamlit run app.py
   ```

---

## 🔬 Model Technical Summary

*   **Algorithm**: Soft-voting Ensemble (`VotingClassifier`) of tuned XGBoost, LightGBM, and CatBoost models.
*   **Feature Space**: 39 features, including 9 advanced engineered HR interaction features (Loyalty, Burnout, Stagnation ratios, Satisfactions).
*   **Evaluation Metrics**: Test set ROC-AUC: **`0.76329`** | Mean 5-Fold CV ROC-AUC: **`0.79558`**.
*   **Data Resampling**: SMOTE applied to training set to balance target distributions (986 non-attrition vs 986 attrition synthetic records).
