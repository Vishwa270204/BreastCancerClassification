import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Survival Predictor",
    page_icon="🎗️",
    layout="wide",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; }
.stApp { background: linear-gradient(135deg, #fdf6f0 0%, #fce8e8 100%); }
div[data-testid="stSidebar"] { background: #1a0a0a; }
div[data-testid="stSidebar"] * { color: white !important; }
.header-card {
    background: linear-gradient(135deg, #8b1a1a, #c0392b);
    border-radius: 16px; padding: 32px 40px; color: white;
    margin-bottom: 28px; box-shadow: 0 8px 32px rgba(139,26,26,0.3);
}
.header-card h1 { font-size: 2.2rem; margin: 0 0 8px 0; color: white; }
.header-card p  { font-size: 1rem; opacity: 0.85; margin: 0; }
.result-alive {
    background: linear-gradient(135deg, #1a5c2a, #27ae60);
    border-radius: 14px; padding: 28px 36px; color: white;
    text-align: center; box-shadow: 0 6px 24px rgba(39,174,96,0.3);
}
.result-dead {
    background: linear-gradient(135deg, #8b1a1a, #c0392b);
    border-radius: 14px; padding: 28px 36px; color: white;
    text-align: center; box-shadow: 0 6px 24px rgba(192,57,43,0.3);
}
.result-alive h2, .result-dead h2 { font-size: 2rem; margin: 0 0 8px 0; color: white; }
.result-alive p,  .result-dead p  { font-size: 1.05rem; opacity: 0.9; margin: 0; }
.prob-card {
    background: white; border-radius: 12px; padding: 18px 24px;
    margin-top: 14px; box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    text-align: center; font-size: 1rem; color: #333;
}
.model-badge {
    display: inline-block; background: #8b1a1a; color: white;
    border-radius: 20px; padding: 4px 14px; font-size: 0.82rem;
    font-weight: 600; margin-bottom: 12px;
}
.info-box {
    background: white; border-left: 4px solid #c0392b;
    border-radius: 8px; padding: 14px 18px; margin-top: 12px;
    font-size: 0.88rem; color: #333; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.section-title {
    font-family: 'DM Serif Display', serif; font-size: 1.2rem;
    color: #8b1a1a; margin: 20px 0 10px 0;
    border-bottom: 1px solid #f0d0d0; padding-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LOAD PKL
# ─────────────────────────────────────────────
PKL_PATH = "breast_cancer_models_compressed.pkl"

@st.cache_resource
def load_bundle(path):
    return joblib.load(path)

if not os.path.exists(PKL_PATH):
    st.error(f"❌ `{PKL_PATH}` not found. Place it in the same folder as app.py.")
    st.stop()

bundle        = load_bundle(PKL_PATH)
scaler        = bundle["scaler"]
features      = bundle["features"]
label_encoder = bundle["label_encoder"]

MODEL_NAMES   = [
    "Logistic Regression", "KNN", "Random Forest", "Decision Tree",
    "SVM", "Gradient Boosting", "Naive Bayes", "XGBoost"
]
SCALED_MODELS = {"Logistic Regression", "KNN", "SVM", "Naive Bayes"}

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-card">
    <h1>🎗️ Breast Cancer Survival Predictor</h1>
    <p>Enter patient details in the sidebar, choose a model, and click <strong>Predict</strong>.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Model")
    selected_model = st.selectbox("Select Model", MODEL_NAMES)

    st.markdown("---")
    st.markdown("## 🧬 Patient Details")

    survival_months = st.number_input("Survival Months", min_value=1, max_value=120, value=40, step=1)
    tumor_size      = st.number_input("Tumor Size (mm)", min_value=1, max_value=200, value=25, step=1)
    reginol_node_positive  = st.number_input("Reginol Node Positive",  min_value=0, max_value=50, value=2, step=1)
    regional_node_examined = st.number_input("Regional Node Examined", min_value=1, max_value=60, value=10, step=1)
    estrogen     = st.selectbox("Estrogen Status",     ["Positive", "Negative"])
    progesterone = st.selectbox("Progesterone Status", ["Positive", "Negative"])
    a_stage      = st.selectbox("A Stage",             ["Regional", "Distant"])

    predict_btn = st.button("🔍 Predict", use_container_width=True)

# ─────────────────────────────────────────────
#  FEATURE VECTOR  — matches features list exactly
# ─────────────────────────────────────────────
def build_input():
    node_positive_ratio = reginol_node_positive / max(regional_node_examined, 1)
    row = {
        "Survival Months":     survival_months,
        "Node_Positive_Ratio": node_positive_ratio,
        "Estrogen Status":     0 if estrogen == "Positive" else 1,
        "Progesterone Status": 0 if progesterone == "Positive" else 1,
        "Tumor Size":          tumor_size,
        "A Stage":             0 if a_stage == "Regional" else 1,
    }
    return pd.DataFrame([row])[features]

# ─────────────────────────────────────────────
#  LAYOUT
# ─────────────────────────────────────────────
col1, col2 = st.columns([1.5, 1], gap="large")

with col1:
    st.markdown('<div class="section-title">📋 Patient Summary</div>', unsafe_allow_html=True)

    node_ratio = reginol_node_positive / max(regional_node_examined, 1)

    # All values as strings to avoid Arrow mixed-type error
    summary = pd.DataFrame({
        "Feature": [
            "Survival Months", "Tumor Size (mm)", "Node Positive",
            "Nodes Examined", "Node Positive Ratio",
            "Estrogen Status", "Progesterone Status", "A Stage"
        ],
        "Value": [
            str(survival_months), str(tumor_size),
            str(reginol_node_positive), str(regional_node_examined),
            f"{node_ratio:.4f}",
            estrogen, progesterone, a_stage
        ]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.caption(f"Features used by model: {', '.join(features)}")

with col2:
    st.markdown('<div class="section-title">🤖 Prediction Result</div>', unsafe_allow_html=True)

    if predict_btn:
        model  = bundle[selected_model]
        X      = build_input()
        X_input = scaler.transform(X) if selected_model in SCALED_MODELS else X.values

        pred_encoded = model.predict(X_input)[0]
        pred_label   = label_encoder.inverse_transform([pred_encoded])[0]

        if pred_label == "Alive":
            st.markdown(f"""
            <div class="result-alive">
                <div class="model-badge">{selected_model}</div>
                <h2>✅ Alive</h2>
                <p>The model predicts the patient is likely to survive.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-dead">
                <div class="model-badge">{selected_model}</div>
                <h2>⚠️ Deceased</h2>
                <p>The model predicts a higher risk of mortality.</p>
            </div>""", unsafe_allow_html=True)

        if hasattr(model, "predict_proba"):
            proba      = model.predict_proba(X_input)[0]
            confidence = proba[pred_encoded] * 100
            alive_p    = proba[0] * 100
            dead_p     = proba[1] * 100
            st.markdown(f"""
            <div class="prob-card">
                <strong>Confidence: {confidence:.1f}%</strong><br>
                <span style="color:#27ae60">Alive: {alive_p:.1f}%</span> &nbsp;|&nbsp;
                <span style="color:#c0392b">Dead: {dead_p:.1f}%</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
            ⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only
            and is not a substitute for clinical diagnosis.
        </div>""", unsafe_allow_html=True)
    else:
        st.info("👈 Fill in patient details in the sidebar and click **Predict**.")
