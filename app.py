from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "churn_ann_model.keras"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "feature_columns.json"


# Page config
st.set_page_config(
    page_title="Churn Predictor Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0e1726 0%, #172554 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: 0 20px 40px rgba(14, 23, 38, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 70%);
        transform: rotate(45deg);
        pointer-events: none;
    }

    .main-header h1 {
        margin: 0;
        font-size: 3.5rem;
        font-weight: 700;
        letter-spacing: -1px;
    }
    
    .main-header p {
        margin-top: 1rem;
        color: #93c5fd;
        font-size: 1.25rem;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    .prediction-container {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.4);
        border-radius: 24px;
        padding: 2.5rem;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.05);
        margin-top: 2rem;
    }
    
    .metric-card {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03);
        border: 1px solid #f8fafc;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 25px rgba(0,0,0,0.08);
    }
    
    .metric-title {
        color: #64748b;
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.8rem;
    }
    
    .metric-value-high {
        font-size: 2.8rem;
        font-weight: 700;
        color: #dc2626;
    }
    
    .metric-value-low {
        font-size: 2.8rem;
        font-weight: 700;
        color: #16a34a;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        height: 3.5rem;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.4);
        background: linear-gradient(135deg, #1d4ed8, #1e40af);
        color: white;
    }
    
    .stProgress > div > div > div > div {
        background-color: #3b82f6;
    }
    
    .dashboard-placeholder {
        text-align: center;
        padding: 5rem 2rem;
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 24px;
        border: 2px dashed #cbd5e1;
        color: #64748b;
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.02);
    }
    
    .dashboard-placeholder h2 {
        color: #1e293b;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
    }
    
    </style>
    """,
    unsafe_allow_html=True,
)


# 🎯 Main Header
st.markdown(
    """
    <div class="main-header">
        <h1>🔍 Churn Predictor </h1>
        <p>Advanced AI-powered customer retention analysis dashboard</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# Load artifacts
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model file: {MODEL_PATH}. Run model.ipynb first."
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Missing scaler file: {SCALER_PATH}. Run model.ipynb first."
        )

    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(
            f"Missing feature columns file: {FEATURE_COLUMNS_PATH}. Run model.ipynb first."
        )

    model = load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    return model, scaler, feature_columns


try:
    model, scaler, feature_columns = load_artifacts()
except Exception as error:
    st.error(str(error))
    st.stop()


# 📊 Sidebar for Inputs
st.sidebar.markdown(
    "<h2>📝 Customer Profile</h2>", 
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("### 🌍 Demographics")
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"], label_visibility="collapsed")
    gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
    age = st.slider("Age (Years)", 18, 100, 40)
    
    st.markdown("<hr style='margin:1rem 0; opacity:0.2;'>", unsafe_allow_html=True)
    st.markdown("### 💰 Financials")
    credit_score = st.slider("Credit Score", 300, 850, 600)
    balance = st.number_input("Account Balance ($)", 0.0, 250000.0, 60000.0, step=1000.0)
    estimated_salary = st.number_input("Estimated Salary ($)", 0.0, 500000.0, 80000.0, step=1000.0)
    
    st.markdown("<hr style='margin:1rem 0; opacity:0.2;'>", unsafe_allow_html=True)
    st.markdown("### ⚡ Engagement & Status")
    tenure = st.slider("Tenure (Years with bank)", 0, 10, 3)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    
    col_cb1, col_cb2 = st.columns(2)
    with col_cb1:
        has_cr_card = st.checkbox("Credit Card", value=True)
    with col_cb2:
        is_active_member = st.checkbox("Active Member", value=True)
        
    st.markdown("<br/>", unsafe_allow_html=True)
    analyze_button = st.button("🔍 Analyze Risk Insights", use_container_width=True)


# 🔮 Prediction Section
if analyze_button:
    raw_row = pd.DataFrame(
        [
            {
                "CreditScore": credit_score,
                "Geography": geography,
                "Gender": gender,
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": num_products,
                "HasCrCard": 1 if has_cr_card else 0,
                "IsActiveMember": 1 if is_active_member else 0,
                "EstimatedSalary": estimated_salary,
            }
        ]
    )

    processed_row = raw_row.copy()
    processed_row["Gender"] = processed_row["Gender"].map({"Female": 0, "Male": 1})
    processed_row = pd.get_dummies(processed_row, columns=["Geography"], drop_first=True)
    processed_row = processed_row.reindex(columns=feature_columns, fill_value=0)

    scaled_row = scaler.transform(processed_row)
    churn_probability = float(model.predict(scaled_row, verbose=0)[0][0])
    stay_probability = 1.0 - churn_probability
    
    is_high_risk = churn_probability >= 0.5
    churn_label = "HIGH RISK" if is_high_risk else "LOW RISK"
    val_class = "metric-value-high" if is_high_risk else "metric-value-low"

    st.markdown('<div class="prediction-container">', unsafe_allow_html=True)

    st.markdown("<h3>📊 AI Prediction Results</h3>", unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">⚠️ Churn Risk</div>
            <div class="{val_class}">{churn_probability * 100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">✅ Retention Rate</div>
            <div class="metric-value-low" style="color: {'#dc2626' if is_high_risk else '#16a34a'};">{stay_probability * 100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🔍 Status</div>
            <div class="{val_class}" style="font-size: 1.8rem; margin-top: 0.5rem;">{churn_label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    
    # Progress bar
    st.progress(float(churn_probability))
    st.caption("AI Model Confidence & Risk Visualizer")
    
    st.markdown("<br/>", unsafe_allow_html=True)

    # Additional insights
    if is_high_risk:
        st.error("🚨 **CRITICAL ALERT:** This customer is at high risk of churning. We strongly recommend immediate intervention offering premium retention strategies or loyalty perks.")
    else:
        st.success("🎉 **SAFE:** This customer shows very low churn risk. Continue maintaining excellent service to preserve this customer satisfaction!")

    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown(
        """
        <div class="dashboard-placeholder">
            <h2>👋 Welcome to Churn Predictor Pro</h2>
            <p style="font-size: 1.25rem; max-width: 600px; margin: 0 auto; line-height: 1.6; opacity: 0.85;">
                Our deeply trained Artificial Neural Network instantly analyzes distinct banking data points to accurately predict the likelihood of customer churn. 
                <br/><br/>
                Adjust the parameters in the demographic and financial profile on the sidebar, and trigger the analysis to uncover AI-driven insights.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )