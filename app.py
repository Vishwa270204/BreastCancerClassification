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
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
}

.stApp {
    background: linear-gradient(135deg, #eff6ff 0%, #ffffff 100%);
}

div[data-testid="stSidebar"] {
    background: #0f172a;
} 

div[data-testid="stSidebar"] * {
    color: white !important;
}

.header-card {
    background: linear-gradient(135deg, #1e40af, #2563eb);
    border-radius: 16px;
    padding: 22px 30px;
    color: white;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(37,99,235,0.25);
}

.header-card h1 {
    font-size: 1.7rem;
    margin: 0 0 6px 0;
    color: white;
}

.header-card p {
    font-size: 0.95rem;
    opacity: 0.9;
    margin: 0;
}

.result-alive,
.result-dead {
    background: linear-gradient(135deg, #1e40af, #2563eb);
    border-radius: 14px;
    padding: 24px 32px;
    color: white;
    text-align: center;
    box-shadow: 0 6px 24px rgba(37,99,235,0.25);
}

.result-alive h2,
.result-dead h2 {
    font-size: 1.8rem;
    margin: 0 0 8px 0;
    color: white;
}

.prob-card {
    background: white;
    border-radius: 12px;
    padding: 18px 24px;
    margin-top: 14px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    text-align: center;
}

.model-badge {
    display: inline-block;
    background: #1e40af;
    color: white;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 12px;
}

.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem;
    color: #1e40af;
    margin: 20px 0 10px 0;
    border-bottom: 1px solid #bfdbfe;
    padding-bottom: 6px;
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
    <p>Enter patient details in the sidebar, choose a model, and click <strong>Predict</strong>.</p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("## ⚙️ Model")
    selected_model = st.selectbox("Select Model", MODEL_NAMES)
    st.markdown("---")
    st.markdown("## 🧬 Patient Details")

    survival_months    = st.number_input("Survival Months", min_value=1, max_value=120, value=40, step=1)
    tumor_size         = st.number_input("Tumor Size (mm)", min_value=1, max_value=200, value=25, step=1)
    reginol_node_pos   = st.number_input("Reginol Node Positive", min_value=0, max_value=50, value=2, step=1)
    regional_node_exam = st.number_input("Regional Node Examined", min_value=1, max_value=60, value=10, step=1)
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

# ── CHART 1: Confidence Bar (Alive vs Dead) ──
def chart_confidence(alive_p, dead_p, model_name):
    fig, ax = plt.subplots(figsize=(8, 3))

    fig.patch.set_facecolor("#eef6ff")
    ax.set_facecolor("#eef6ff")

    bars = ax.barh(
        ["Dead", "Alive"],
        [dead_p, alive_p],
        color=["#93c5fd", "#2563eb"]
    )

    for bar, val in zip(bars, [dead_p, alive_p]):
        ax.text(
            val + 1,
            bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%",
            va="center",
            fontweight="bold"
        )

    ax.set_xlim(0, 115)
    ax.set_xlabel("Probability (%)")
    ax.set_title(
        f"Prediction Confidence — {model_name}",
        color="#1e40af",
        fontweight="bold"
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
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
            results.append({"Model": name, "Alive %": 100 if pred == 0 else 0,
                            "Dead %": 100 if pred == 1 else 0})

    df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(6, 2))
    fig.patch.set_facecolor("#fdf6f0")
    ax.set_facecolor("#fdf6f0")

    x = np.arange(len(df))
    w = 0.38
    ax.bar(x - w/2, df["Alive %"], width=w, color="#27ae60", label="Alive", edgecolor="none")
    ax.bar(x + w/2, df["Dead %"],  width=w, color="#c0392b", label="Dead",  edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=30, ha="right", fontsize=8, color="#333")
    ax.set_ylabel("Probability (%)", fontsize=9, color="#555")
    ax.set_title("All Models — Alive vs Dead Probability", fontsize=11,
                 color="#8b1a1a", fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9)
    ax.tick_params(colors="#555")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # highlight selected model
    idx = MODEL_NAMES.index(selected_model)
    ax.axvspan(idx - 0.5, idx + 0.5, color="#8b1a1a", alpha=0.07, zorder=0)

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
    st.caption(f"Features used by model: {', '.join(features)}")

with col2:
    st.markdown('<div class="section-title">🤖 Prediction Result</div>', unsafe_allow_html=True)

    if predict_btn:
        model   = bundle[selected_model]
        X       = build_input()
        X_input = scaler.transform(X) if selected_model in SCALED_MODELS else X.values

        pred_encoded = model.predict(X_input)[0]
        pred_label   = label_encoder.inverse_transform([pred_encoded])[0]

        if pred_label == "Alive":
            st.markdown(f"""<div class="result-alive">
                <div class="model-badge">{selected_model}</div>
                <h2>Alive</h2>
                <p>The model predicts the patient is likely to survive.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="result-dead">
                <div class="model-badge">{selected_model}</div>
                <h2>Dead</h2>
                <p>The model predicts a higher risk of mortality.</p>
            </div>""", unsafe_allow_html=True)

        alive_p, dead_p = 50.0, 50.0
        if hasattr(model, "predict_proba"):
            proba       = model.predict_proba(X_input)[0]
            confidence  = proba[pred_encoded] * 100
            alive_p     = proba[0] * 100
            dead_p      = proba[1] * 100
            st.markdown(f"""<div class="prob-card">
                <strong>Confidence: {confidence:.1f}%</strong><br>
                <span style="color:#27ae60">Alive: {alive_p:.1f}%</span> &nbsp;|&nbsp;
                <span style="color:#c0392b">Dead: {dead_p:.1f}%</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("👈 Fill in patient details in the sidebar and click **Predict**.")
# =========================
# CHARTS BELOW SUMMARY
# =========================

if predict_btn:

    st.markdown("---")

    st.markdown(
        '<div class="section-title">📊 Prediction Confidence</div>',
        unsafe_allow_html=True
    )
    st.pyplot(chart_confidence(alive_p, dead_p, selected_model))

    st.markdown(
        '<div class="section-title">📊 All Models Comparison</div>',
        unsafe_allow_html=True
    )
    st.pyplot(chart_model_comparison())
