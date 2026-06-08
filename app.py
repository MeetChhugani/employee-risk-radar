import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from groq import Groq

# --- Load Environment Variables ---
load_dotenv()

# --- Load API Key (Identical to original app logic) ---
api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")

# --- Page Config ---
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👥",
    layout="wide",  # Wide layout for dashboard-style columns
    initial_sidebar_state="expanded"
)

# --- Inject Premium Custom CSS & Fonts ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Apply Font */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Plus Jakarta Sans', 'Outfit', sans-serif !important;
        color: #E2E8F0 !important;
    }
    
    /* Background Gradient */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #090D1A 0%, #02040A 100%) !important;
    }
    
    /* Hide Default Header & Footer Decoration */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Block container styling */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1250px !important;
    }
    
    /* Custom Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #05070C !important;
        border-right: 1px solid rgba(255, 255, 255, 0.04) !important;
    }
    
    /* Glassmorphic Header Card with Animated Radial Glow */
    .header-card {
        background: linear-gradient(135deg, rgba(20, 30, 55, 0.4) 0%, rgba(10, 15, 30, 0.25) 100%);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 24px;
        padding: 35px 20px;
        text-align: center;
        backdrop-filter: blur(25px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
    }
    
    .header-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.07) 0%, transparent 60%);
        pointer-events: none;
    }
    
    .header-card h1 {
        font-size: 3rem;
        font-weight: 800;
        margin: 0 0 10px 0;
        background: linear-gradient(90deg, #6366F1 0%, #A855F7 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1.5px;
    }
    
    .header-card p {
        font-size: 1.15rem;
        color: #94A3B8;
        margin: 0;
        font-weight: 400;
    }
    
    /* Styled Section Subheaders */
    .section-header {
        background: linear-gradient(90deg, rgba(99, 102, 241, 0.12) 0%, rgba(99, 102, 241, 0) 100%);
        border-left: 4px solid #6366F1;
        padding: 10px 20px;
        border-radius: 0 12px 12px 0;
        margin: 24px 0 18px 0;
        font-weight: 700;
        font-size: 1.35rem;
        color: #F1F5F9;
        letter-spacing: -0.5px;
    }
    
    /* Custom Glass Cards with Hover Scale & Glowing borders */
    .glass-card {
        background: rgba(10, 15, 30, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 24px;
        backdrop-filter: blur(20px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35), inset 0 1px 1px rgba(255, 255, 255, 0.04);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.25);
        box-shadow: 0 15px 35px rgba(99, 102, 241, 0.08);
    }
    
    /* Custom Risk Status Glows */
    .gauge-card-green {
        border-color: rgba(16, 185, 129, 0.2) !important;
        box-shadow: 0 15px 40px rgba(16, 185, 129, 0.08), inset 0 1px 1px rgba(255, 255, 255, 0.04) !important;
    }
    .gauge-card-red {
        border-color: rgba(244, 63, 94, 0.2) !important;
        box-shadow: 0 15px 40px rgba(244, 63, 94, 0.08), inset 0 1px 1px rgba(255, 255, 255, 0.04) !important;
    }
    
    /* Custom CSS styled Inputs & Dropdowns */
    div[data-baseweb="select"], div[data-baseweb="input"], input {
        background-color: rgba(6, 8, 16, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        color: #F8FAFC !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-baseweb="select"]:hover, div[data-baseweb="input"]:hover, input:hover {
        border-color: rgba(99, 102, 241, 0.45) !important;
        background-color: rgba(10, 15, 30, 0.9) !important;
        box-shadow: 0 0 10px rgba(99, 102, 241, 0.15) !important;
    }
    
    /* Custom Segmented Pill-styled Tabs */
    div[data-testid="stTabBar"] {
        background: rgba(6, 8, 16, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        padding: 6px !important;
        margin-bottom: 24px !important;
    }
    
    div[data-testid="stTabBar"] button {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #94A3B8 !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 8px 16px !important;
        margin-right: 4px !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    div[data-testid="stTabBar"] button:hover {
        color: #FFFFFF !important;
        background: rgba(255, 255, 255, 0.04) !important;
    }
    
    div[data-testid="stTabBar"] button[aria-selected="true"] {
        color: #FFFFFF !important;
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25) !important;
    }
    
    /* Hide Default Tab Selection Indicator Line */
    div[data-testid="stTabBar"] div[role="tablist"] + div {
        display: none !important;
    }
    
    /* Predict Button Override */
    div.stButton > button {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #DB2777 100%) !important;
        color: #FFFFFF !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1.15rem !important;
        border: none !important;
        padding: 14px 28px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 18px rgba(99, 102, 241, 0.35) !important;
        letter-spacing: -0.2px;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 24px rgba(99, 102, 241, 0.5) !important;
        background: linear-gradient(135deg, #4338CA 0%, #6D28D9 50%, #BE185D 100%) !important;
    }
    
    /* Gauge Indicator styling */
    .gauge-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        margin: 15px 0;
    }
    
    .gauge-value {
        font-size: 4.8rem;
        font-weight: 800;
        line-height: 1;
        margin: 8px 0;
        letter-spacing: -2px;
    }
    
    .gauge-label {
        font-size: 0.85rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #94A3B8;
        font-weight: 700;
    }
    
    /* Custom Progress Bar */
    .custom-progress {
        width: 100%;
        background-color: rgba(255, 255, 255, 0.04);
        border-radius: 20px;
        height: 12px;
        margin: 15px 0;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .custom-progress-bar {
        height: 100%;
        border-radius: 20px;
        transition: width 0.8s cubic-bezier(0.1, 0.8, 0.3, 1);
    }
    
    /* Custom Styling Overrides for Native Chat Messages */
    div[data-testid="stChatMessage"] {
        background-color: rgba(15, 23, 42, 0.45) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 16px !important;
        padding: 14px 18px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15) !important;
    }
    
    div[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
        color: #E2E8F0 !important;
        line-height: 1.6 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Model Artifact ---
try:
    artifact = joblib.load("models/attrition_model.pkl")
    model = artifact["model"]
    features = artifact["features"]
    threshold = artifact["threshold"]
except Exception as e:
    st.error(f"❌ Error loading model file models/attrition_model.pkl. Ensure it exists. Error: {e}")
    st.stop()

# --- Preset Profiles logic ---
presets = {
    "Select Preset Profile...": None,
    "🔴 High-Risk Software Engineer": {
        "age": 25, "monthly_income": 3500, "job_satisfaction": 1, "work_life_balance": 1, "years_at_company": 2,
        "years_in_current_role": 2, "overtime": 1, "distance_from_home": 24, "num_companies_worked": 4,
        "total_working_years": 4, "years_since_promotion": 0, "years_with_manager": 1,
        "environment_satisfaction": 1, "stock_option_level": 0, "department": 1, "job_level": 1,
        "job_involvement": 2, "marital_status": 2, "business_travel": 1, "education": 2, "gender": 1,
        "percent_salary_hike": 11, "training_times": 2, "education_field": 1, "job_role": 6,
        "relationship_satisfaction": 2, "daily_rate": 500, "hourly_rate": 45, "monthly_rate": 12000,
        "performance_rating": 3
    },
    "🟢 Stable Executive Manager": {
        "age": 48, "monthly_income": 18000, "job_satisfaction": 4, "work_life_balance": 3, "years_at_company": 15,
        "years_in_current_role": 8, "overtime": 0, "distance_from_home": 2, "num_companies_worked": 1,
        "total_working_years": 25, "years_since_promotion": 2, "years_with_manager": 7,
        "environment_satisfaction": 4, "stock_option_level": 2, "department": 1, "job_level": 5,
        "job_involvement": 4, "marital_status": 1, "business_travel": 0, "education": 4, "gender": 0,
        "percent_salary_hike": 18, "training_times": 3, "education_field": 3, "job_role": 3,
        "relationship_satisfaction": 4, "daily_rate": 1200, "hourly_rate": 85, "monthly_rate": 22000,
        "performance_rating": 4
    },
    "🟡 Overworked Sales Rep": {
        "age": 31, "monthly_income": 4200, "job_satisfaction": 2, "work_life_balance": 2, "years_at_company": 4,
        "years_in_current_role": 3, "overtime": 1, "distance_from_home": 15, "num_companies_worked": 3,
        "total_working_years": 8, "years_since_promotion": 1, "years_with_manager": 2,
        "environment_satisfaction": 2, "stock_option_level": 0, "department": 2, "job_level": 2,
        "job_involvement": 3, "marital_status": 2, "business_travel": 1, "education": 3, "gender": 1,
        "percent_salary_hike": 12, "training_times": 1, "education_field": 2, "job_role": 8,
        "relationship_satisfaction": 3, "daily_rate": 700, "hourly_rate": 60, "monthly_rate": 15000,
        "performance_rating": 3
    }
}

st.sidebar.markdown("<div class='section-header'>⚡ Load Preset Profile</div>", unsafe_allow_html=True)
selected_preset = st.sidebar.selectbox("Choose a template:", list(presets.keys()))

# Initialize session state keys
inputs_to_initialize = {
    "age": 30, "monthly_income": 5000, "job_satisfaction": 3, "work_life_balance": 3, "years_at_company": 3,
    "years_in_current_role": 2, "overtime": 0, "distance_from_home": 10, "num_companies_worked": 2,
    "total_working_years": 8, "years_since_promotion": 1, "years_with_manager": 3,
    "environment_satisfaction": 3, "stock_option_level": 1, "department": 1, "job_level": 2,
    "job_involvement": 3, "marital_status": 1, "business_travel": 2, "education": 3, "gender": 1,
    "percent_salary_hike": 12, "training_times": 2, "education_field": 1, "job_role": 6,
    "relationship_satisfaction": 3, "daily_rate": 800, "hourly_rate": 65, "monthly_rate": 14000,
    "performance_rating": 3
}

# Apply preset selection
if selected_preset != "Select Preset Profile..." and presets[selected_preset] is not None:
    preset_vals = presets[selected_preset]
    for k, v in preset_vals.items():
        st.session_state[k] = v
    st.session_state['prediction_made'] = False
    st.session_state['auto_suggested'] = False
    st.session_state['messages'] = []
    st.sidebar.success(f"Loaded {selected_preset}!")

for k, v in inputs_to_initialize.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --- Groq LLM Client helper ---
def call_llm(prompt, api_key, max_tokens=500):
    if not api_key:
        return "⚠️ Please configure your Groq API Key in your environment variables or Streamlit secrets to enable AI suggestions."
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI assistant is currently unavailable. Details: {str(e)[:100]}"

# --- Main Page UI ---
st.markdown("""
    <div class='header-card'>
        <h1>👥 Employee Risk Radar</h1>
        <p>AI-powered Attrition Risk Predictor &amp; Generative AI Retention Advisor</p>
    </div>
""", unsafe_allow_html=True)

form_col, dashboard_col = st.columns([1.1, 0.9])

with form_col:
    st.markdown("<div class='section-header'>📝 Employee Demographics &amp; Role</div>", unsafe_allow_html=True)
    
    # Wrap tabs into glass-card
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    tab_gen, tab_satisfaction, tab_comp = st.tabs(["📋 General & Role", "💡 Satisfaction & Environment", "💰 Compensation & Performance"])
    
    with tab_gen:
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            age = st.number_input("Age", min_value=18, max_value=60, key="age")
            gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", key="gender")
            marital_status = st.selectbox("Marital Status", [0, 1, 2], format_func=lambda x: {0:"Divorced", 1:"Married", 2:"Single"}[x], key="marital_status")
            education = st.selectbox("Education Level", [1, 2, 3, 4, 5], format_func=lambda x: {1:"Below College", 2:"College", 3:"Bachelor", 4:"Master", 5:"Doctor"}[x], key="education")
            education_field = st.selectbox("Education Field", [0, 1, 2, 3, 4, 5], format_func=lambda x: {0:"Human Resources", 1:"Life Sciences", 2:"Marketing", 3:"Medical", 4:"Other", 5:"Technical Degree"}[x], key="education_field")
        with col_g2:
            department = st.selectbox("Department", [0, 1, 2], format_func=lambda x: {0:"Human Resources", 1:"Research & Development", 2:"Sales"}[x], key="department")
            job_role = st.selectbox("Job Role", list(range(9)), format_func=lambda x: {0:"Healthcare Rep", 1:"Human Resources", 2:"Lab Technician", 3:"Manager", 4:"Manufacturing Director", 5:"Research Director", 6:"Research Scientist", 7:"Sales Executive", 8:"Sales Rep"}[x], key="job_role")
            job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5], key="job_level")
            
            # --- FIX: Alphabetical class mapping (Non-Travel=0, Travel_Frequently=1, Travel_Rarely=2) ---
            business_travel = st.selectbox(
                "Business Travel",
                [0, 1, 2],
                format_func=lambda x: {0:"Non-Travel", 1:"Travel Frequently", 2:"Travel Rarely"}[x],
                key="business_travel"
            )
            
            distance_from_home = st.number_input("Distance From Home (km)", min_value=1, max_value=30, key="distance_from_home")

    with tab_satisfaction:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4], format_func=lambda x: {1:"Low", 2:"Medium", 3:"High", 4:"Very High"}[x], key="job_satisfaction")
            environment_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4], format_func=lambda x: {1:"Low", 2:"Medium", 3:"High", 4:"Very High"}[x], key="environment_satisfaction")
            relationship_satisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4], format_func=lambda x: {1:"Low", 2:"Medium", 3:"High", 4:"Very High"}[x], key="relationship_satisfaction")
            job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4], format_func=lambda x: {1:"Low", 2:"Medium", 3:"High", 4:"Very High"}[x], key="job_involvement")
        with col_s2:
            work_life_balance = st.selectbox("Work Life Balance", [1, 2, 3, 4], format_func=lambda x: {1:"Bad", 2:"Good", 3:"Better", 4:"Best"}[x], key="work_life_balance")
            overtime = st.selectbox("OverTime Requirement", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="overtime")
            training_times = st.number_input("Training Times Last Year", min_value=0, max_value=6, key="training_times")
            stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3], key="stock_option_level")

    with tab_comp:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=20000, step=500, key="monthly_income")
            percent_salary_hike = st.number_input("Percent Salary Hike (%)", min_value=0, max_value=25, key="percent_salary_hike")
            performance_rating = st.selectbox("Performance Rating", [3, 4], format_func=lambda x: "Excellent" if x == 3 else "Outstanding", key="performance_rating")
            num_companies_worked = st.number_input("Companies Worked Prior", min_value=0, max_value=10, key="num_companies_worked")
            total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, key="total_working_years")
        with col_c2:
            years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, key="years_at_company")
            years_in_current_role = st.number_input("Years in Current Role", min_value=0, max_value=20, key="years_in_current_role")
            years_since_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, key="years_since_promotion")
            years_with_manager = st.number_input("Years With Current Manager", min_value=0, max_value=20, key="years_with_manager")
            
            daily_rate = st.number_input("Daily Rate ($)", min_value=100, max_value=1500, key="daily_rate")
            hourly_rate = st.number_input("Hourly Rate ($)", min_value=30, max_value=100, key="hourly_rate")
            monthly_rate = st.number_input("Monthly Rate ($)", min_value=2000, max_value=27000, key="monthly_rate")
    st.markdown("</div>", unsafe_allow_html=True) # Close glass-card

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🔍 Analyze Attrition Risk", use_container_width=True)

# Generate Input Dictionary matching XGBoost feature signatures
input_dict = {
    "Age": st.session_state["age"], "BusinessTravel": st.session_state["business_travel"], "DailyRate": st.session_state["daily_rate"],
    "Department": st.session_state["department"], "DistanceFromHome": st.session_state["distance_from_home"], "Education": st.session_state["education"],
    "EducationField": st.session_state["education_field"], "EnvironmentSatisfaction": st.session_state["environment_satisfaction"], "Gender": st.session_state["gender"],
    "HourlyRate": st.session_state["hourly_rate"], "JobInvolvement": st.session_state["job_involvement"], "JobLevel": st.session_state["job_level"],
    "JobRole": st.session_state["job_role"], "JobSatisfaction": st.session_state["job_satisfaction"], "MaritalStatus": st.session_state["marital_status"],
    "MonthlyIncome": st.session_state["monthly_income"], "MonthlyRate": st.session_state["monthly_rate"], "NumCompaniesWorked": st.session_state["num_companies_worked"],
    "OverTime": st.session_state["overtime"], "PercentSalaryHike": st.session_state["percent_salary_hike"], "PerformanceRating": st.session_state["performance_rating"],
    "RelationshipSatisfaction": st.session_state["relationship_satisfaction"], "StockOptionLevel": st.session_state["stock_option_level"], "TotalWorkingYears": st.session_state["total_working_years"],
    "TrainingTimesLastYear": st.session_state["training_times"], "WorkLifeBalance": st.session_state["work_life_balance"], "YearsAtCompany": st.session_state["years_at_company"],
    "YearsInCurrentRole": st.session_state["years_in_current_role"], "YearsSinceLastPromotion": st.session_state["years_since_promotion"], "YearsWithCurrManager": st.session_state["years_with_manager"]
}

if predict_clicked:
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=features, fill_value=0)

    prob = model.predict_proba(input_df)[0][1]
    prediction = prob >= threshold

    st.session_state['employee_data'] = input_dict
    st.session_state['attrition_prob'] = prob
    st.session_state['prediction'] = prediction
    st.session_state['prediction_made'] = True
    st.session_state['auto_suggested'] = False
    st.session_state['messages'] = []

# --- Right Column: Results Dashboard ---
with dashboard_col:
    st.markdown("<div class='section-header'>📊 Attrition Risk Output</div>", unsafe_allow_html=True)
    
    if st.session_state.get('prediction_made'):
        prob = st.session_state['attrition_prob']
        prediction = st.session_state['prediction']
        
        status_color = "#F43F5E" if prediction else "#10B981"
        status_text = "⚠️ HIGH ATTRITION RISK" if prediction else "✅ LOW ATTRITION RISK"
        card_class = "gauge-card-red" if prediction else "gauge-card-green"
        glow_color = "244, 63, 94" if prediction else "16, 185, 129"
        
        # Circular-style gauge and clean meter bar with dynamic glowing cards
        st.markdown(f"""
            <div class='glass-card {card_class}'>
                <div class='gauge-container'>
                    <div class='gauge-label'>Probability of Departure</div>
                    <div class='gauge-value' style='color: {status_color}; text-shadow: 0 0 15px rgba({glow_color}, 0.3); font-weight: 800;'>
                        {prob * 100:.1f}%
                    </div>
                    <div style='font-size: 1.15rem; font-weight: 600; color: {status_color}; margin-top: 5px;'>
                        {status_text}
                    </div>
                </div>
                <div class='custom-progress'>
                    <div class='custom-progress-bar' style='width: {prob * 100}%; background: linear-gradient(90deg, #8B5CF6 0%, {status_color} 100%);'></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Explanations using Matplotlib & SHAP Styled transparently
        with st.expander("🔍 Feature Impact Analysis (SHAP)", expanded=True):
            explainer = shap.TreeExplainer(model)
            input_df = pd.DataFrame([st.session_state['employee_data']])
            input_df = input_df.reindex(columns=features, fill_value=0)
            shap_values = explainer.shap_values(input_df)

            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(10, 5), facecolor='none')
            ax.set_facecolor('none')
            
            shap.plots.bar(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=explainer.expected_value,
                    data=input_df.iloc[0],
                    feature_names=input_df.columns.tolist()
                ),
                max_display=8,
                show=False
            )
            
            plt.gca().patch.set_alpha(0.0)
            fig.patch.set_alpha(0.0)
            ax.tick_params(colors='#94A3B8', labelsize=11)
            ax.xaxis.label.set_color('#E2E8F0')
            ax.yaxis.label.set_color('#E2E8F0')
            plt.title("Key Factors Contributing to Attrition Score", color='#F1F5F9', fontsize=12, pad=10, fontweight='semibold')
            plt.tight_layout()
            
            st.pyplot(fig, clear_figure=True)
            plt.close()

        # --- Relocated Chatbot Panel inside Right Dashboard Column for zero-scroll visibility ---
        st.markdown("<div class='section-header'>🤖 Conversational HR Retention Advisor</div>", unsafe_allow_html=True)
        
        if not st.session_state.get('auto_suggested'):
            employee = st.session_state['employee_data']
            prob = st.session_state['attrition_prob']
            
            if prob < threshold:
                st.session_state['messages'].append({
                    "role": "assistant",
                    "content": "✅ **Stable Risk Profile**: This employee currently demonstrates low attrition potential. Keep regular feedback cycles active to preserve role alignment."
                })
                st.session_state['auto_suggested'] = True
            else:
                dept_map = {0: "Human Resources", 1: "Research & Development", 2: "Sales"}
                role_map = {0: "Healthcare Rep", 1: "Human Resources", 2: "Lab Technician", 3: "Manager", 
                            4: "Manufacturing Director", 5: "Research Director", 6: "Research Scientist", 
                            7: "Sales Executive", 8: "Sales Rep"}
                satisfaction_map = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
                
                auto_prompt = f"""You are an expert HR consultant. An employee has been flagged with {prob*100:.1f}% attrition risk (Threshold is {threshold*100:.0f}%).
                
                Employee Profile:
                - Age: {employee['Age']}
                - Department: {dept_map[employee['Department']]}
                - Job Role: {role_map[employee['JobRole']]}
                - Monthly Income: ${employee['MonthlyIncome']}
                - Job Satisfaction: {satisfaction_map[employee['JobSatisfaction']]}
                - Work Life Balance: {satisfaction_map[employee['WorkLifeBalance']]}
                - OverTime Requirement: {'Yes' if employee['OverTime']==1 else 'No'}
                - Years at Company: {employee['YearsAtCompany']}
                - Years Since Last Promotion: {employee['YearsSinceLastPromotion']}
                - Environment Satisfaction: {satisfaction_map[employee['EnvironmentSatisfaction']]}
                - Distance From Home: {employee['DistanceFromHome']} km
                
                Provide:
                1. Top 3 primary triggers why this employee might resign.
                2. Top 4 actionable, specific retention protocols.
                3. Instant prioritization goal for this week.
                
                Format response with markdown bolding, clear spacing, and keep it crisp and business-focused."""
                
                with st.spinner("Analyzing employee risk profile and building retention guidelines..."):
                    auto_reply = call_llm(auto_prompt, api_key, max_tokens=600)
                
                st.session_state['messages'].append({"role": "assistant", "content": auto_reply})
                st.session_state['auto_suggested'] = True

        # Display styled messages using native Streamlit chat bubbles styled by our CSS rules
        for msg in st.session_state['messages']:
            with st.chat_message(msg['role']):
                st.write(msg['content'])

        # Column-embedded Chat Input
        if user_input := st.chat_input("Query specialized retention strategies...", key="dashboard_chat_input"):
            st.session_state['messages'].append({"role": "user", "content": user_input})
            
            employee = st.session_state['employee_data']
            prob = st.session_state['attrition_prob']
            employee_context = f"Employee has a {prob*100:.1f}% attrition risk. Income is ${employee['MonthlyIncome']}/mo. Overtime: {'Yes' if employee['OverTime']==1 else 'No'}. Job satisfaction: {employee['JobSatisfaction']}/4."
            
            chat_prompt = f"""You are an expert HR consultant. Context: {employee_context}
            
            Question: {user_input}
            
            Provide highly practical advice in 3-4 professional, actionable sentences."""
            
            with st.spinner("AI consultant is formulating response..."):
                reply = call_llm(chat_prompt, api_key, max_tokens=300)
                
            st.session_state['messages'].append({"role": "assistant", "content": reply})
            st.rerun()
            
    else:
        st.markdown("""
            <div class='glass-card' style='text-align: center; padding: 40px; color: #64748B;'>
                <span style='font-size: 3rem;'>📊</span>
                <h3 style='margin: 15px 0 10px 0; color: #94A3B8;'>Awaiting Employee Assessment</h3>
                <p>Provide employee specifications on the left and click 'Analyze' to render risk analytics and activate AI assistant.</p>
            </div>
        """, unsafe_allow_html=True)