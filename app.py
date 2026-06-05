import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(page_title="Breast Cancer Survival Predictor", page_icon="🎗️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Nunito:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif;
}

/* ── App Background ── */
.stApp {
    background: #f9fafb;
}

/* ── Sidebar ── */
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a3c1a 0%, #2d5a2d 60%, #1f3f1f 100%);
    border-right: 3px solid #4caf50;
}

div[data-testid="stSidebar"] * {
    color: #e8f5e9 !important;
}

div[data-testid="stSidebar"] .stSelectbox label,
div[data-testid="stSidebar"] .stNumberInput label {
    color: #a5d6a7 !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
    color: #c8e6c9 !important;
}

/* Sidebar input controls */
div[data-testid="stSidebar"] input,
div[data-testid="stSidebar"] select,
div[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.1) !important;
    border: 1px solid rgba(76, 175, 80, 0.4) !important;
    color: white !important;
    border-radius: 8px !important;
}

/* Predict button */
div[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #c62828, #e53935) !important;
    color: white !important;
    border: none !important;
    border-radius: 30px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.6rem 1.2rem !important;
    box-shadow: 0 4px 18px rgba(198,40,40,0.4) !important;
    transition: all 0.25s ease !important;
}

div[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, #b71c1c, #c62828) !important;
    box-shadow: 0 6px 24px rgba(183,28,28,0.5) !important;
    transform: translateY(-1px);
}

/* ── Header Banner ── */
.header-card {
    background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 40%, #c62828 100%);
    border-radius: 16px;
    padding: 20px 32px;
    margin-bottom: 18px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.12);
    position: relative;
    overflow: hidden;
}

.header-card::before {
    content: "🎗️";
    position: absolute;
    right: 24px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 3.5rem;
    opacity: 0.25;
}

.header-card h1 {
    font-size: 1.5rem;
    margin: 0 0 4px 0;
    color: white;
    font-family: 'Playfair Display', serif;
}

.header-card p {
    font-size: 0.92rem;
    color: rgba(255,255,255,0.85);
    margin: 0;
}

/* ── Section Titles ── */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    color: #1b5e20;
    margin: 18px 0 10px 0;
    border-bottom: 2px solid #a5d6a7;
    padding-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ── Result Cards ── */
.result-alive {
    background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
    border: 2px solid #4caf50;
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 18px rgba(76,175,80,0.18);
}

.result-alive h2 {
    color: #1b5e20;
    font-size: 1.6rem;
    margin: 6px 0 4px;
}

.result-alive p {
    color: #388e3c;
    font-size: 0.9rem;
    margin: 0;
}

.result-dead {
    background: linear-gradient(135deg, #ffebee, #fce4ec);
    border: 2px solid #e53935;
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 18px rgba(229,57,53,0.18);
}

.result-dead h2 {
    color: #b71c1c;
    font-size: 1.6rem;
    margin: 6px 0 4px;
}

.result-dead p {
    color: #c62828;
    font-size: 0.9rem;
    margin: 0;
}

/* ── Model Badge ── */
.model-badge {
    display: inline-block;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.badge-alive {
    background: #1b5e20;
    color: white;
}

.badge-dead {
    background: #c62828;
    color: white;
}

/* ── Probability Card ── */
.prob-card {
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 14px 20px;
    margin-top: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    text-align: center;
    font-size: 0.95rem;
}

.prob-card strong {
    color: #263238;
    font-size: 1rem;
}

/* ── Divider ── */
hr {
    border: none;
    border-top: 2px dashed #c8e6c9;
    margin: 18px 0;
}

/* ── Dataframe Styling ── */
.stDataFrame {
    border-radius: 10px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
}

/* ── Info Box ── */
.stAlert {
    border-radius: 10px !important;
    border-left: 4px solid #4caf50 !important;
    background: #f1f8e9 !important;
}
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
    <h1>🎗️ Breast Cancer Survival Predictor</h1>
    <p>Enter patient details in the sidebar, choose a model, and click <strong>Predict</strong> to assess survival likelihood.</p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("## ⚙️ Model")
    selected_model = st.selectbox("Select Model", MODEL_NAMES)
    st.markdown("---")
    st.markdown("## 🧬 Patient Details")

    survival_months    = st.number_input("Survival Months",        min_value=1,  max_value=120, value=40, step=1)
    tumor_size         = st.number_input("Tumor Size (mm)",         min_value=1,  max_value=200, value=25, step=1)
    reginol_node_pos   = st.number_input("Reginol Node Positive",   min_value=0,  max_value=50,  value=2,  step=1)
    regional_node_exam = st.number_input("Regional Node Examined",  min_value=1,  max_value=60,  value=10, step=1)
    estrogen           = st.selectbox("Estrogen Status",            ["Positive", "Negative"])
    progesterone       = st.selectbox("Progesterone Status",        ["Positive", "Negative"])
    a_stage            = st.selectbox("A Stage",                    ["Regional", "Distant"])

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

def chart_confidence(alive_p, dead_p, model_name):
    fig, ax = plt.subplots(figsize=(6, 2.2))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    colors = ["#e53935", "#43a047"]
    bars = ax.barh(
        ["Dead", "Alive"],
        [dead_p, alive_p],
        color=colors,
        height=0.5,
        edgecolor="none"
    )

    for bar, val in zip(bars, [dead_p, alive_p]):
        ax.text(
            val + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#263238"
        )

    ax.set_xlim(0, 120)
    ax.set_xlabel("Probability (%)", fontsize=8, color="#607d8b")
    ax.set_title(f"Confidence — {model_name}", fontsize=10, color="#1b5e20", fontweight="bold", pad=10)
    ax.tick_params(labelsize=9, colors="#455a64")

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.xaxis.set_tick_params(color="#cfd8dc")
    ax.yaxis.set_tick_params(color="#cfd8dc")
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

    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafafa")

    x = np.arange(len(df))
    w = 0.35

    ax.bar(x - w/2, df["Alive %"], width=w, color="#43a047", label="Alive", edgecolor="none", alpha=0.9)
    ax.bar(x + w/2, df["Dead %"],  width=w, color="#e53935", label="Dead",  edgecolor="none", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=28, ha="right", fontsize=7.5, color="#455a64")
    ax.set_ylabel("Probability (%)", fontsize=8, color="#607d8b")
    ax.set_title("All Models — Side-by-Side Comparison", fontsize=10, color="#1b5e20", fontweight="bold", pad=10)
    ax.set_ylim(0, 118)

    ax.legend(fontsize=8, framealpha=0.6, edgecolor="#e0e0e0")

    # Highlight selected model
    idx = MODEL_NAMES.index(selected_model)
    ax.axvspan(idx - 0.5, idx + 0.5, color="#e8f5e9", alpha=0.7, zorder=0)
    ax.annotate("▲ selected", xy=(idx, 105), ha="center", fontsize=7, color="#2e7d32", fontweight="600")

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.yaxis.grid(True, color="#eeeeee", linewidth=0.8)
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
            <div class="result-alive">
                <span class="model-badge badge-alive">{selected_model}</span>
                <h2>✅ Alive</h2>
                <p>The model predicts the patient is likely to survive.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-dead">
                <span class="model-badge badge-dead">{selected_model}</span>
                <h2>⚠️ High Risk</h2>
                <p>The model predicts a higher risk of mortality.</p>
            </div>""", unsafe_allow_html=True)

        alive_p, dead_p = 50.0, 50.0
        if hasattr(model, "predict_proba"):
            proba      = model.predict_proba(X_input)[0]
            confidence = proba[pred_encoded] * 100
            alive_p    = proba[0] * 100
            dead_p     = proba[1] * 100
            st.markdown(f"""
            <div class="prob-card">
                <strong>Confidence: {confidence:.1f}%</strong><br>
                <span style="color:#2e7d32; font-weight:600;">🟢 Alive: {alive_p:.1f}%</span>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                <span style="color:#c62828; font-weight:600;">🔴 Dead: {dead_p:.1f}%</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("👈 Fill in patient details in the sidebar and click **Predict**.")

# ── CHARTS ──
if predict_btn:
    st.markdown("---")
    st.markdown('<div class="section-title">📊 Prediction Confidence</div>', unsafe_allow_html=True)
    st.pyplot(chart_confidence(alive_p, dead_p, selected_model))

    st.markdown('<div class="section-title">📊 All Models Comparison</div>', unsafe_allow_html=True)
    st.pyplot(chart_model_comparison())
