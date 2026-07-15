# 👥 Employee Risk Radar

An AI-powered HR analytics platform designed to predict employee attrition risks, explain model decisions using Explainable AI (SHAP), and generate tailored retention strategies using a Conversational LLM Advisor.

---

## 📊 Key Highlights & Metrics

*   **XGBoost Risk Classifier**: Trained on IBM Watson's HR Attrition dataset (1,470 records, 30 features) using extreme gradient boosting.
*   **Imbalance Correction & Optimization**: Resolves severe class imbalance (84% retention vs 16% attrition) using **SMOTE** (Synthetic Minority Over-sampling Technique) and optimizes decision boundaries at a custom threshold of **`0.30`** (maximizing recall for at-risk cases).
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

*   **Algorithm**: XGBoost Classifier (`n_estimators=300`, `max_depth=6`, `learning_rate=0.1`)
*   **Evaluation Metric**: ROC-AUC: **`0.7704`**
*   **Data Resampling**: SMOTE applied to training set to balance target distributions (986 non-attrition vs 986 attrition synthetic records).
