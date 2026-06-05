import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Breast Cancer Survival Predictor",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Outfit:wght@300;400;500;600&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}
h1, h2, h3 { font-family: 'Playfair Display', serif; }

.stApp {
    background: #f7f0ee;
    padding-top: 0 !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 0 !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 100% !important;
}

/* ── Fixed top header bar ── */
.top-header {
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 999;
    background: linear-gradient(90deg, #6b0f1a 0%, #b91c1c 60%, #dc2626 100%);
    padding: 10px 28px;
    display: flex;
    align-items: center;
    gap: 14px;
    box-shadow: 0 2px 16px rgba(107,15,26,0.35);
}
.top-header .ribbon { font-size: 1.5rem; }
.top-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 1.25rem;
    color: #fff;
    margin: 0;
    letter-spacing: 0.01em;
}
.top-header p {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.75);
    margin: 0;
}
.header-divider {
    width: 1px; height: 36px;
    background: rgba(255,255,255,0.25);
    margin: 0 6px;
}

/* Push content below fixed header */
.main-content-spacer { margin-top: 72px; }

/* ── Force sidebar always visible, hide toggle ── */
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"] {
    min-width: 270px !important;
    max-width: 270px !important;
    width: 270px !important;
    top: 52px !important;        /* below fixed header */
    background: #1c0608 !important;
    border-right: 1px solid #3d1010;
    position: fixed !important;
    height: calc(100vh - 52px) !important;
    overflow-y: auto;
}
section[data-testid="stSidebar"] * { color: #f5e6e6 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] h2 { color: #f9d0d0 !important; }
section[data-testid="stSidebar"] hr { border-color: #3d1010 !important; }
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #b91c1c, #dc2626) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px !important;
    width: 100% !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em;
    box-shadow: 0 4px 14px rgba(185,28,28,0.4) !important;
    transition: opacity 0.2s;
}
section[data-testid="stSidebar"] .stButton > button:hover { opacity: 0.88 !important; }

/* Offset main area for sidebar */
.main .block-container {
    margin-left: 280px !important;
}

/* ── Cards ── */
.result-alive {
    background: linear-gradient(135deg, #14532d, #16a34a);
    border-radius: 12px; padding: 22px 28px; color: white;
    text-align: center; box-shadow: 0 6px 20px rgba(22,163,74,0.28);
}
.result-dead {
    background: linear-gradient(135deg, #6b0f1a, #b91c1c);
    border-radius: 12px; padding: 22px 28px; color: white;
    text-align: center; box-shadow: 0 6px 20px rgba(185,28,28,0.28);
}
.result-alive h2, .result-dead h2 {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem; margin: 0 0 6px 0; color: white;
}
.result-alive p, .result-dead p { font-size: 0.9rem; opacity: 0.88; margin: 0; }

.prob-card {
    background: white; border-radius: 10px; padding: 14px 20px;
    margin-top: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    text-align: center; font-size: 0.92rem; color: #333;
}
.model-badge {
    display: inline-block; background: rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.35);
    color: white; border-radius: 20px;
    padding: 2px 12px; font-size: 0.75rem;
    font-weight: 600; margin-bottom: 8px;
    letter-spacing: 0.04em; text-transform: uppercase;
}
.info-box {
    background: #fff5f5; border-left: 3px solid #b91c1c;
    border-radius: 6px; padding: 10px 14px; margin-top: 10px;
    font-size: 0.8rem; color: #555;
}
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem; color: #6b0f1a;
    margin: 16px 0 8px 0;
    border-bottom: 1px solid #e8cece; padding-bottom: 5px;
}
</style>
""", unsafe_allow_html=True)

# ── FIXED HEADER ──
st.markdown("""
<div class="top-header">
    <span class="ribbon">🎗️</span>
    <div>
        <h1>Breast Cancer Survival Predictor</h1>
        <p>ML-powered risk assessment · For educational use only</p>
    </div>
    <div class="header-divider"></div>
    <p style="color:rgba(255,255,255,0.6);font-size:0.75rem;margin:0;">
        8 models &nbsp;·&nbsp; Logistic Regression, KNN, RF, DT, SVM, GB, NB, XGBoost
    </p>
</div>
<div class="main-content-spacer"></div>
""", unsafe_allow_html=True)

# ── LOAD PKL ──
PKL_PATH = "breast_cancer_models.pkl"

@st.cache_resource
def load_bundle(path):
    return joblib.load(path)

if not os.path.exists(PKL_PATH):
    st.error(f"❌ `{PKL_PATH}` not found. Place it in the same directory as this script.")
    st.stop()

bundle        = load_bundle(PKL_PATH)
scaler        = bundle["scaler"]
features      = bundle["features"]
label_encoder = bundle["label_encoder"]

MODEL_NAMES   = ["Logistic Regression", "KNN", "Random Forest", "Decision Tree",
                 "SVM", "Gradient Boosting", "Naive Bayes", "XGBoost"]
SCALED_MODELS = {"Logistic Regression", "KNN", "SVM", "Naive Bayes"}

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("## ⚙️ Model")
    selected_model = st.selectbox("Select Model", MODEL_NAMES)
    st.markdown("---")
    st.markdown("## 🧬 Patient Details")

    survival_months    = st.number_input("Survival Months",          min_value=1,  max_value=120, value=40,  step=1)
    tumor_size         = st.number_input("Tumor Size (mm)",          min_value=1,  max_value=200, value=25,  step=1)
    reginol_node_pos   = st.number_input("Reginol Node Positive",    min_value=0,  max_value=50,  value=2,   step=1)
    regional_node_exam = st.number_input("Regional Node Examined",   min_value=1,  max_value=60,  value=10,  step=1)
    estrogen           = st.selectbox("Estrogen Status",             ["Positive", "Negative"])
    progesterone       = st.selectbox("Progesterone Status",         ["Positive", "Negative"])
    a_stage            = st.selectbox("A Stage",                     ["Regional", "Distant"])

    predict_btn = st.button("🔍 Predict Survival", use_container_width=True)

# ── FEATURE VECTOR ──
def build_input():
    node_ratio = reginol_node_pos / max(regional_node_exam, 1)
    mapping = {
        "Survival Months":        survival_months,
        "Tumor Size":             tumor_size,
        "Reginol Node Positive":  reginol_node_pos,
        "Regional Node Examined": regional_node_exam,
        "Node_Positive_Ratio":    node_ratio,
        "Estrogen Status":        0 if estrogen == "Positive" else 1,
        "Progesterone Status":    0 if progesterone == "Positive" else 1,
        "A Stage":                0 if a_stage == "Regional" else 1,
    }
    return pd.DataFrame([{f: mapping[f] for f in features}])[features]

def get_input_for_model(name):
    X = build_input()
    return scaler.transform(X) if name in SCALED_MODELS else X.values

# ── CHART 1: Confidence Bar ──
def chart_confidence(alive_p, dead_p, model_name):
    fig, ax = plt.subplots(figsize=(4.5, 2.4))
    fig.patch.set_facecolor("#f7f0ee")
    ax.set_facecolor("#f7f0ee")
    bars = ax.barh(["Deceased", "Alive"], [dead_p, alive_p],
                   color=["#b91c1c", "#16a34a"], height=0.45, edgecolor="none")
    for bar, val in zip(bars, [dead_p, alive_p]):
        ax.text(val + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10, fontweight="bold", color="#333")
    ax.set_xlim(0, 118)
    ax.set_xlabel("Probability (%)", fontsize=8, color="#777")
    ax.set_title(f"Confidence · {model_name}", fontsize=9, color="#6b0f1a", fontweight="bold")
    ax.tick_params(colors="#555", labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout(pad=0.5)
    return fig

# ── CHART 2: All Models Comparison ──
def chart_model_comparison():
    results = []
    for name in MODEL_NAMES:
        m = bundle[name]
        Xi = get_input_for_model(name)
        if hasattr(m, "predict_proba"):
            p = m.predict_proba(Xi)[0]
            results.append({"Model": name, "Alive %": p[0]*100, "Dead %": p[1]*100})
        else:
            pred = m.predict(Xi)[0]
            results.append({"Model": name,
                             "Alive %": 100 if pred == 0 else 0,
                             "Dead %":  100 if pred == 1 else 0})

    df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(6, 2.8))
    fig.patch.set_facecolor("#f7f0ee")
    ax.set_facecolor("#f7f0ee")

    x = np.arange(len(df))
    w = 0.36
    ax.bar(x - w/2, df["Alive %"], width=w, color="#16a34a", label="Alive", edgecolor="none")
    ax.bar(x + w/2, df["Dead %"],  width=w, color="#b91c1c", label="Dead",  edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [n.replace(" ", "\n") for n in df["Model"]],
        fontsize=6.5, color="#333"
    )
    ax.set_ylabel("Probability (%)", fontsize=8, color="#777")
    ax.set_title("All Models — Alive vs Dead", fontsize=9, color="#6b0f1a", fontweight="bold")
    ax.set_ylim(0, 118)
    ax.legend(fontsize=8, framealpha=0)
    ax.tick_params(colors="#555", labelsize=7)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Highlight selected model
    idx = MODEL_NAMES.index(selected_model)
    ax.axvspan(idx - 0.5, idx + 0.5, color="#6b0f1a", alpha=0.08, zorder=0)

    plt.tight_layout(pad=0.5)
    return fig

# ── TOP ROW: Patient Summary | Prediction Result ──
col1, col2 = st.columns([1.3, 1], gap="large")

node_ratio = reginol_node_pos / max(regional_node_exam, 1)

with col1:
    st.markdown('<div class="section-title">📋 Patient Summary</div>', unsafe_allow_html=True)
    summary = pd.DataFrame({
        "Feature": ["Survival Months", "Tumor Size (mm)", "Reginol Node Positive",
                    "Regional Node Examined", "Node Positive Ratio",
                    "Estrogen Status", "Progesterone Status", "A Stage"],
        "Value":   [str(survival_months), str(tumor_size), str(reginol_node_pos),
                    str(regional_node_exam), f"{node_ratio:.4f}",
                    estrogen, progesterone, a_stage]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.caption(f"Model features: {', '.join(features)}")

with col2:
    st.markdown('<div class="section-title">🤖 Prediction Result</div>', unsafe_allow_html=True)

    if predict_btn:
        model   = bundle[selected_model]
        X_input = get_input_for_model(selected_model)

        pred_encoded = model.predict(X_input)[0]
        pred_label   = label_encoder.inverse_transform([pred_encoded])[0]

        if pred_label == "Alive":
            st.markdown(f"""<div class="result-alive">
                <div class="model-badge">{selected_model}</div>
                <h2>✅ Alive</h2>
                <p>The model predicts the patient is likely to survive.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="result-dead">
                <div class="model-badge">{selected_model}</div>
                <h2>⚠️ Deceased</h2>
                <p>The model predicts a higher risk of mortality.</p>
            </div>""", unsafe_allow_html=True)

        alive_p, dead_p = 50.0, 50.0
        if hasattr(model, "predict_proba"):
            proba      = model.predict_proba(X_input)[0]
            confidence = proba[pred_encoded] * 100
            alive_p    = proba[0] * 100
            dead_p     = proba[1] * 100
            st.markdown(f"""<div class="prob-card">
                <strong>Confidence: {confidence:.1f}%</strong><br>
                <span style="color:#16a34a">Alive: {alive_p:.1f}%</span> &nbsp;|&nbsp;
                <span style="color:#b91c1c">Deceased: {dead_p:.1f}%</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("""<div class="info-box">
            ⚠️ <strong>Disclaimer:</strong> For educational purposes only.
            Not a substitute for clinical diagnosis.
        </div>""", unsafe_allow_html=True)

    else:
        st.info("👈 Fill in patient details in the sidebar and click **Predict Survival**.")

# ── BOTTOM ROW: Both Charts Side by Side (only after prediction) ──
if predict_btn:
    st.markdown('<div class="section-title">📊 Analytics</div>', unsafe_allow_html=True)
    chart_col1, chart_col2 = st.columns([1, 1.4], gap="large")

    with chart_col1:
        st.pyplot(chart_confidence(alive_p, dead_p, selected_model))

    with chart_col2:
        st.pyplot(chart_model_comparison())
