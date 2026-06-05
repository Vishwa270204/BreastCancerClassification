import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Breast Cancer Survival Predictor", page_icon="🎗️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Fraunces', serif; letter-spacing: -0.01em; }

/* ── App background ── */
.stApp {
    background:
        radial-gradient(1000px 500px at 90% -10%, #fde2e7 0%, transparent 60%),
        radial-gradient(900px 500px at -10% 110%, #fbe9d6 0%, transparent 60%),
        #fbf7f4;
}

/* ── Sidebar ── */
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1b2e 0%, #2a1f3d 50%, #3a1f3a 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
div[data-testid="stSidebar"] * { color: #ece6f5 !important; }
div[data-testid="stSidebar"] h2 {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    color: #fff !important;
    letter-spacing: -0.01em;
}
div[data-testid="stSidebar"] label {
    color: #c9b8e0 !important;
    font-weight: 500 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
div[data-testid="stSidebar"] input,
div[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #fff !important;
    border-radius: 12px !important;
}
div[data-testid="stSidebar"] hr { border-top: 1px solid rgba(255,255,255,0.08); }

/* Predict button */
div[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #ff8aa3, #ec5e7b) !important;
    color: white !important;
    border: none !important;
    border-radius: 999px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.7rem 1.2rem !important;
    box-shadow: 0 10px 30px -10px rgba(236,94,123,0.6) !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 14px 36px -10px rgba(236,94,123,0.75) !important;
}

/* ── Header ── */
.header-card {
    background: linear-gradient(120deg, #ffffff 0%, #fff5f3 60%, #ffe6ec 100%);
    border: 1px solid rgba(236,94,123,0.15);
    border-radius: 22px;
    padding: 26px 32px;
    margin-bottom: 22px;
    box-shadow: 0 20px 40px -28px rgba(58,31,58,0.25);
    position: relative;
    overflow: hidden;
}
.header-card::after {
    content: "";
    position: absolute; inset: 0;
    background: radial-gradient(400px 200px at 100% 0%, rgba(255,138,163,0.18), transparent 60%);
    pointer-events: none;
}
.header-eyebrow {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(236,94,123,0.1);
    color: #b03654;
    font-size: 0.72rem; font-weight: 600;
    padding: 4px 12px; border-radius: 999px;
    text-transform: uppercase; letter-spacing: 0.12em;
    margin-bottom: 10px;
}
.header-card h1 {
    font-size: 2rem; margin: 0 0 6px 0;
    color: #2a1f3d; font-weight: 600;
}
.header-card p {
    font-size: 0.95rem; color: #6b5d7a; margin: 0; max-width: 640px;
}

/* ── Section title ── */
.section-title {
    font-family: 'Fraunces', serif;
    font-size: 1.05rem; font-weight: 600;
    color: #2a1f3d;
    margin: 8px 0 12px 0;
    display: flex; align-items: center; gap: 8px;
}
.section-title::before {
    content: ""; width: 4px; height: 18px; border-radius: 4px;
    background: linear-gradient(180deg, #ec5e7b, #ff8aa3);
}

/* ── Result cards ── */
.result-card {
    border-radius: 18px;
    padding: 22px 24px;
    text-align: center;
    border: 1px solid;
    position: relative;
    overflow: hidden;
}
.result-alive {
    background: linear-gradient(135deg, #ffffff, #f0fbf4);
    border-color: #b6e5c6;
    box-shadow: 0 18px 40px -24px rgba(46,125,50,0.35);
}
.result-alive h2 { color: #1f6b3a; font-size: 1.5rem; margin: 6px 0 4px; font-weight: 600; }
.result-alive p  { color: #4a8a5e; font-size: 0.88rem; margin: 0; }

.result-dead {
    background: linear-gradient(135deg, #ffffff, #fff0f1);
    border-color: #f5c2c8;
    box-shadow: 0 18px 40px -24px rgba(198,40,40,0.35);
}
.result-dead h2 { color: #a02338; font-size: 1.5rem; margin: 6px 0 4px; font-weight: 600; }
.result-dead p  { color: #b85462; font-size: 0.88rem; margin: 0; }

.model-badge {
    display: inline-block;
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 0.7rem; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    margin-bottom: 6px;
}
.badge-alive { background: #1f6b3a; color: #e8f6ec; }
.badge-dead  { background: #a02338; color: #ffe8ec; }

/* ── Probability card ── */
.prob-card {
    background: white;
    border: 1px solid #efe6ec;
    border-radius: 16px;
    padding: 16px 20px;
    margin-top: 14px;
    box-shadow: 0 10px 30px -20px rgba(58,31,58,0.2);
    text-align: center;
    font-size: 0.92rem;
    color: #4a3f5c;
}
.prob-card strong { color: #2a1f3d; font-weight: 600; }

/* ── Dataframe ── */
.stDataFrame {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid #efe6ec !important;
    box-shadow: 0 10px 30px -22px rgba(58,31,58,0.18);
}

/* ── Info alert ── */
.stAlert {
    border-radius: 14px !important;
    border: 1px solid #f5d6dd !important;
    background: #fff5f6 !important;
    color: #6b5d7a !important;
}

hr { border: none; border-top: 1px solid #efe6ec; margin: 22px 0; }

/* Caption */
.stCaption, [data-testid="stCaptionContainer"] { color: #8b7f99 !important; }
</style>
""", unsafe_allow_html=True)

# ── LOAD PKL ──
PKL_PATH = "breast_cancer_models.pkl"

@st.cache_resource
def load_bundle(path):
    return joblib.load(path)

if not os.path.exists(PKL_PATH):
    st.error(f"❌ `{PKL_PATH}` not found.")
    st.stop()

bundle        = load_bundle(PKL_PATH)
scaler        = bundle["scaler"]
features      = bundle["features"]
label_encoder = bundle["label_encoder"]

MODEL_NAMES   = ["Logistic Regression", "KNN", "Random Forest", "Decision Tree",
                 "SVM", "Gradient Boosting", "Naive Bayes", "XGBoost"]
SCALED_MODELS = {"Logistic Regression", "KNN", "SVM", "Naive Bayes"}

# ── HEADER ──
st.markdown("""
<div class="header-card">
    <span class="header-eyebrow">🎗️ Clinical ML Assistant</span>
    <h1>Breast Cancer Survival Predictor</h1>
    <p>Enter patient details in the sidebar, choose a model, and click <strong>Predict</strong> to assess survival likelihood across eight trained classifiers.</p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("## ⚙️ Model")
    selected_model = st.selectbox("Select Model", MODEL_NAMES)
    st.markdown("---")
    st.markdown("## 🧬 Patient Details")

    survival_months    = st.number_input("Survival Months",         min_value=1, max_value=120, value=40, step=1)
    tumor_size         = st.number_input("Tumor Size (mm)",         min_value=1, max_value=200, value=25, step=1)
    reginol_node_pos   = st.number_input("Reginol Node Positive",   min_value=0, max_value=50,  value=2,  step=1)
    regional_node_exam = st.number_input("Regional Node Examined",  min_value=1, max_value=60,  value=10, step=1)
    estrogen           = st.selectbox("Estrogen Status",     ["Positive", "Negative"])
    progesterone       = st.selectbox("Progesterone Status", ["Positive", "Negative"])
    a_stage            = st.selectbox("A Stage",             ["Regional", "Distant"])

    predict_btn = st.button("🔍 Predict", use_container_width=True)

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
    row = {f: mapping[f] for f in features}
    return pd.DataFrame([row])[features]

def get_input_for_model(model_name):
    X = build_input()
    return scaler.transform(X) if model_name in SCALED_MODELS else X.values

def _style_axes(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors="#6b5d7a", labelsize=9)

def chart_confidence(alive_p, dead_p, model_name):
    fig, ax = plt.subplots(figsize=(6, 2.4))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    colors = ["#ec5e7b", "#5cb87a"]
    bars = ax.barh(["Dead", "Alive"], [dead_p, alive_p], color=colors, height=0.55, edgecolor="none")

    for bar, val in zip(bars, [dead_p, alive_p]):
        ax.text(val + 1.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%",
                va="center", fontsize=10, fontweight="600", color="#2a1f3d")

    ax.set_xlim(0, 120)
    ax.set_xlabel("Probability (%)", fontsize=8, color="#8b7f99")
    ax.set_title(f"Confidence — {model_name}", fontsize=11, color="#2a1f3d", fontweight="600", pad=12, loc="left")
    _style_axes(ax)
    ax.xaxis.grid(True, color="#f1ebf2", linewidth=0.8)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig

def chart_model_comparison():
    results = []
    for name in MODEL_NAMES:
        m = bundle[name]
        Xi = get_input_for_model(name)
        if hasattr(m, "predict_proba"):
            p = m.predict_proba(Xi)[0]
            results.append({"Model": name, "Alive %": p[0] * 100, "Dead %": p[1] * 100})
        else:
            pred = m.predict(Xi)[0]
            results.append({"Model": name, "Alive %": 100 if pred == 0 else 0, "Dead %": 100 if pred == 1 else 0})
    df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    x = np.arange(len(df))
    w = 0.36
    ax.bar(x - w/2, df["Alive %"], width=w, color="#5cb87a", label="Alive", edgecolor="none")
    ax.bar(x + w/2, df["Dead %"],  width=w, color="#ec5e7b", label="Dead",  edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=25, ha="right", fontsize=8, color="#6b5d7a")
    ax.set_ylabel("Probability (%)", fontsize=8, color="#8b7f99")
    ax.set_title("All Models — Side-by-Side Comparison", fontsize=11,
                 color="#2a1f3d", fontweight="600", pad=12, loc="left")
    ax.set_ylim(0, 118)
    ax.legend(fontsize=8, framealpha=0, loc="upper right")

    idx = MODEL_NAMES.index(selected_model)
    ax.axvspan(idx - 0.5, idx + 0.5, color="#fff0f3", alpha=0.9, zorder=0)
    ax.annotate("▲ selected", xy=(idx, 108), ha="center", fontsize=8,
                color="#b03654", fontweight="600")

    _style_axes(ax)
    ax.yaxis.grid(True, color="#f1ebf2", linewidth=0.8)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig

# ── LAYOUT ──
col1, col2 = st.columns([1.5, 1], gap="large")

with col1:
    st.markdown('<div class="section-title">📋 Patient Summary</div>', unsafe_allow_html=True)
    node_ratio = reginol_node_pos / max(regional_node_exam, 1)
    summary = pd.DataFrame({
        "Feature": ["Survival Months", "Tumor Size (mm)", "Reginol Node Positive",
                    "Regional Node Examined", "Node Positive Ratio",
                    "Estrogen Status", "Progesterone Status", "A Stage"],
        "Value":   [str(survival_months), str(tumor_size), str(reginol_node_pos),
                    str(regional_node_exam), f"{node_ratio:.4f}",
                    estrogen, progesterone, a_stage]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.caption(f"🔧 Features used: {', '.join(features)}")

with col2:
    st.markdown('<div class="section-title">🤖 Prediction Result</div>', unsafe_allow_html=True)

    if predict_btn:
        model   = bundle[selected_model]
        X       = build_input()
        X_input = scaler.transform(X) if selected_model in SCALED_MODELS else X.values

        pred_encoded = model.predict(X_input)[0]
        pred_label   = label_encoder.inverse_transform([pred_encoded])[0]

        if pred_label == "Alive":
            st.markdown(f"""
            <div class="result-card result-alive">
                <span class="model-badge badge-alive">{selected_model}</span>
                <h2>✅ Alive</h2>
                <p>The model predicts the patient is likely to survive.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-dead">
                <span class="model-badge badge-dead">{selected_model}</span>
                <h2>⚠️ High Risk</h2>
                <p>The model predicts a higher risk of mortality.</p>
            </div>
            """, unsafe_allow_html=True)

        alive_p, dead_p = 50.0, 50.0
        if hasattr(model, "predict_proba"):
            proba      = model.predict_proba(X_input)[0]
            confidence = proba[pred_encoded] * 100
            alive_p    = proba[0] * 100
            dead_p     = proba[1] * 100
            st.markdown(f"""
            <div class="prob-card">
                <strong>Confidence: {confidence:.1f}%</strong><br>
                🟢 Alive: {alive_p:.1f}%  &nbsp;|&nbsp;  🔴 Dead: {dead_p:.1f}%
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("👈 Fill in patient details in the sidebar and click **Predict**.")

# ── CHARTS ──
if predict_btn:
    st.markdown("---")
    st.markdown('<div class="section-title">📊 Prediction Confidence</div>', unsafe_allow_html=True)
    st.pyplot(chart_confidence(alive_p, dead_p, selected_model))

    st.markdown('<div class="section-title">📊 All Models Comparison</div>', unsafe_allow_html=True)
    st.pyplot(chart_model_comparison())
