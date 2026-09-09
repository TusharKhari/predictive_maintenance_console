import html
from pathlib import Path
import textwrap

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
from sklearn.pipeline import Pipeline
import streamlit as st

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PredixAI | Predictive Maintenance",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DESIGN SYSTEM
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: "Inter", sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 20% 0%, rgba(14, 116, 144, 0.13), transparent 30%),
            linear-gradient(180deg, #07111f 0%, #050b14 45%, #040810 100%);
        color: #e5edf7;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 5rem;
        padding-bottom: 4rem;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #07111e 0%, #050b14 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.10);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.8rem;
    }

    h1, h2, h3, h4 {
        letter-spacing: -0.025em;
    }

    .mono {
        font-family: "JetBrains Mono", monospace;
    }

    .eyebrow {
        color: #38bdf8;
        font-size: 0.70rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }

    .page-title {
        color: #f8fafc;
        font-size: 2.45rem;
        line-height: 1.1;
        font-weight: 800;
        margin: 0;
    }

    .page-subtitle {
        color: #94a3b8;
        font-size: 0.92rem;
        line-height: 1.6;
        margin-top: 0.7rem;
        max-width: 780px;
    }

    .system-status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 7px 11px;
        border-radius: 999px;
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.22);
        color: #6ee7b7;
        font-size: 0.70rem;
        font-weight: 800;
        letter-spacing: 0.08em;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #34d399;
        box-shadow: 0 0 10px rgba(52, 211, 153, 0.8);
    }

    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 2rem;
        margin-bottom: 0.9rem;
        color: #f8fafc;
        font-size: 1rem;
        font-weight: 750;
    }

    .section-line {
        height: 1px;
        flex: 1;
        background: linear-gradient(90deg, rgba(148, 163, 184, 0.16), transparent);
    }

    .card {
        background: linear-gradient(145deg, rgba(15, 27, 45, 0.92), rgba(7, 14, 26, 0.88));
        border: 1px solid rgba(148, 163, 184, 0.11);
        border-radius: 15px;
        padding: 1.15rem 1.25rem;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.18);
        height: 100%;
    }

    .card:hover {
        border-color: rgba(56, 189, 248, 0.22);
    }

    .metric-label {
        color: #7f8ea3;
        font-size: 0.67rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .metric-value {
        color: #f8fafc;
        font-family: "JetBrains Mono", monospace;
        font-size: 1.55rem;
        font-weight: 700;
        margin-top: 0.35rem;
    }

    .metric-description {
        color: #64748b;
        font-size: 0.73rem;
        margin-top: 0.3rem;
    }

    .input-panel {
        background: linear-gradient(145deg, rgba(12, 27, 45, 0.90), rgba(7, 14, 26, 0.92));
        border: 1px solid rgba(56, 189, 248, 0.12);
        border-radius: 16px;
        padding: 1.25rem 1.4rem 0.75rem 1.4rem;
        box-shadow: inset 0 1px rgba(255,255,255,0.025), 0 15px 40px rgba(0,0,0,0.16);
    }

    .input-caption {
        color: #64748b;
        font-size: 0.75rem;
        margin-bottom: 1rem;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 9px;
        border-radius: 999px;
        font-size: 0.65rem;
        font-weight: 800;
        letter-spacing: 0.07em;
    }

    .badge-green {
        color: #6ee7b7;
        background: rgba(16,185,129,0.08);
        border: 1px solid rgba(16,185,129,0.20);
    }

    .badge-yellow {
        color: #fcd34d;
        background: rgba(245,158,11,0.08);
        border: 1px solid rgba(245,158,11,0.20);
    }

    .badge-red {
        color: #fca5a5;
        background: rgba(239,68,68,0.08);
        border: 1px solid rgba(239,68,68,0.20);
    }

    .badge-blue {
        color: #7dd3fc;
        background: rgba(14,165,233,0.08);
        border: 1px solid rgba(14,165,233,0.20);
    }

    .prediction-panel {
        background: linear-gradient(145deg, rgba(12, 24, 40, 0.96), rgba(6, 12, 22, 0.94));
        border-radius: 17px;
        padding: 1.4rem;
        border: 1px solid rgba(148,163,184,0.11);
        min-height: 100%;
    }

    .prediction-danger {
        border-color: rgba(239,68,68,0.28);
        box-shadow: inset 4px 0 #ef4444;
    }

    .prediction-warning {
        border-color: rgba(245,158,11,0.28);
        box-shadow: inset 4px 0 #f59e0b;
    }

    .prediction-success {
        border-color: rgba(16,185,129,0.22);
        box-shadow: inset 4px 0 #10b981;
    }

    .prediction-title {
        color: #f8fafc;
        font-size: 1.25rem;
        font-weight: 800;
        margin-top: 0.75rem;
    }

    .prediction-copy {
        color: #94a3b8;
        font-size: 0.83rem;
        line-height: 1.65;
    }

    .pipeline-card {
        background: rgba(7, 14, 26, 0.72);
        border: 1px solid rgba(148,163,184,0.10);
        border-radius: 14px;
        padding: 1rem;
        text-align: center;
    }

    .pipeline-number {
        color: #38bdf8;
        font-family: "JetBrains Mono", monospace;
        font-size: 0.68rem;
        font-weight: 700;
    }

    .pipeline-title {
        color: #e2e8f0;
        font-size: 0.78rem;
        font-weight: 750;
        margin-top: 0.35rem;
    }

    .pipeline-desc {
        color: #64748b;
        font-size: 0.67rem;
        margin-top: 0.25rem;
    }

    .action-box {
        margin-top: 1rem;
        padding: 0.95rem 1rem;
        border-radius: 12px;
        background: rgba(15,23,42,0.72);
        border: 1px solid rgba(148,163,184,0.09);
    }

    .action-title {
        color: #38bdf8;
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    .action-text {
        color: #cbd5e1;
        font-size: 0.78rem;
        line-height: 1.55;
        margin-top: 0.35rem;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        border: 1px solid rgba(56,189,248,0.30);
        background: linear-gradient(135deg, #0284c7, #0369a1);
        color: white;
        font-weight: 750;
        min-height: 44px;
        box-shadow: 0 8px 20px rgba(2,132,199,0.16);
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        border-color: #38bdf8;
        background: linear-gradient(135deg, #0ea5e9, #0369a1);
        transform: translateY(-1px);
        box-shadow: 0 10px 25px rgba(14,165,233,0.22);
    }

    div[data-testid="stSlider"] label {
        color: #cbd5e1 !important;
        font-size: 0.78rem !important;
        font-weight: 650 !important;
    }

    div[data-testid="stSlider"] div[role="slider"] {
        background: #38bdf8 !important;
        border: 2px solid #0284c7 !important;
        box-shadow: 0 0 8px rgba(56,189,248,0.35) !important;
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stTextInput"] label {
        color: #94a3b8 !important;
        font-size: 0.75rem !important;
        font-weight: 650 !important;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    .footer {
        margin-top: 3rem;
        padding-top: 1.1rem;
        border-top: 1px solid rgba(148,163,184,0.08);
        display: flex;
        justify-content: space-between;
        color: #475569;
        font-size: 0.68rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
# def load_model():
#     base_dir = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
#     candidate_paths = [
#         Path("predictive_maintenance_model.joblib"),
#         base_dir / "ml" / "models" / "predictive_maintenance_model.joblib",
#         base_dir.parent / "ml" / "models" / "predictive_maintenance_model.joblib",
#     ]
def load_model():
    model_path = Path(__file__).resolve().parent / "predictive_maintenance_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at: {model_path}"
        )

    return joblib.load(model_path)

    for path in candidate_paths:
        if path.is_file():
            try:
                return joblib.load(path)
            except Exception as error:
                st.sidebar.error(f"Unable to load model artifact: `{path}`")
                st.sidebar.exception(error)
                return None
    return None



model = load_model()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
            <div style="
                width:34px; height:34px; border-radius:9px;
                display:flex; align-items:center; justify-content:center;
                background:rgba(14,165,233,0.10); border:1px solid rgba(14,165,233,0.22);
                font-size:18px;
            ">⚡</div>
            <div>
                <div style="font-weight:800; font-size:15px; color:#f8fafc; letter-spacing:0.04em;">
                    PREDIXAI
                </div>
                <div style="color:#64748b; font-size:10px; letter-spacing:0.08em;">
                    INDUSTRIAL ML
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("Predictive maintenance & equipment health analytics")
    st.divider()

    st.markdown("##### Equipment Configuration")
    machine_type = st.selectbox(
        "Machine class",
        options=["L", "M", "H"],
        format_func=lambda x: {
            "L": "L — Economy",
            "M": "M — Standard",
            "H": "H — Heavy Duty",
        }[x],
        help="Machine quality class used by the trained model.",
    )

    machine_id = st.text_input(
        "Asset identifier",
        value="CNC-MILL-084B",
    )

    st.divider()
    st.markdown("##### Model Information")
    st.markdown(
        """
        <div style="
            background:rgba(15,23,42,0.55);
            border:1px solid rgba(148,163,184,0.08);
            border-radius:10px;
            padding:11px;
            font-size:12px;
            line-height:1.7;
            color:#94a3b8;
        ">
            <div><span style="color:#64748b;">Algorithm</span><br>
            <b style="color:#e2e8f0;">Gradient Boosted Classifier</b></div>
            <div style="margin-top:8px;">
            <span style="color:#64748b;">Dataset</span><br>
            <b style="color:#e2e8f0;">AI4I 2020 Predictive Maintenance</b>
            </div>
            <div style="margin-top:8px;">
            <span style="color:#64748b;">Inference</span><br>
            <b style="color:#e2e8f0;">Binary failure-risk classification</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown(
        """
        <div style="color:#64748b; font-size:10px; line-height:1.6;">
            <b style="color:#94a3b8;">Engineering note</b><br>
            Model predictions are decision-support signals, not a replacement
            for certified inspection or maintenance procedures.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MODEL CHECK
# ============================================================

if model is None:
    st.error("### Model artifact unavailable")
    st.info(
        "Place the trained pipeline at "
        "`ml/models/predictive_maintenance_model.joblib` "
        "to enable inference."
    )
    st.stop()


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns([4, 1])

with header_left:
    st.markdown("<div class='eyebrow'>Industrial machine learning system</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Predictive Maintenance Console</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="page-subtitle">
            Real-time equipment health assessment using machine telemetry,
            engineered operating features, and a trained classification pipeline.
            Adjust the sensor inputs below and run an inference cycle.
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown(
        f"""
        <div style="text-align:right; padding-top:10px;">
            <div class="system-status">
                <span class="status-dot"></span>
                MODEL ONLINE
            </div>
            <div style="color:#64748b; font-size:11px; margin-top:8px;">
                Asset
                <span class="mono" style="color:#94a3b8;">
                    {html.escape(machine_id)}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# INPUT TELEMETRY
# ============================================================

st.markdown(
    """
    <div class="section-header">
        Sensor Telemetry
        <div class="section-line"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='input-panel'>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="input-caption">
        Configure the current operating state of the target machine.
        Values are passed through the same feature schema expected by the
        serialized ML pipeline.
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)

with c1:
    air_temperature = st.slider(
        "Air temperature",
        250.0,
        350.0,
        300.1,
        0.1,
        format="%.1f K",
        help="Ambient air temperature measured near the machine.",
    )

    process_temperature = st.slider(
        "Process temperature",
        250.0,
        400.0,
        310.2,
        0.1,
        format="%.1f K",
        help="Current process temperature of the equipment.",
    )

with c2:
    rotational_speed = st.slider(
        "Rotational speed",
        500,
        3500,
        1500,
        10,
        format="%d RPM",
        help="Spindle rotational speed.",
    )

    torque = st.slider(
        "Spindle torque",
        0.0,
        120.0,
        42.5,
        0.1,
        format="%.1f Nm",
        help="Applied spindle torque.",
    )

with c3:
    tool_wear = st.slider(
        "Tool wear",
        0,
        300,
        180,
        1,
        format="%d min",
        help="Accumulated tool operating time.",
    )

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    predict_button = st.button("⚡ Run ML Diagnostic", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

temperature_difference = process_temperature - air_temperature
power_proxy = rotational_speed * torque


# ============================================================
# FEATURE STATUS
# ============================================================

wear_status = (
    ("badge-green", "NORMAL")
    if tool_wear < 120
    else (
        ("badge-yellow", "ELEVATED")
        if tool_wear < 200
        else ("badge-red", "HIGH")
    )
)

thermal_status = (
    ("badge-green", "LOW")
    if temperature_difference < 8
    else (
        ("badge-yellow", "MODERATE")
        if temperature_difference < 12
        else ("badge-red", "HIGH")
    )
)

torque_status = (
    ("badge-green", "NORMAL")
    if torque < 50
    else (
        ("badge-yellow", "HEAVY")
        if torque < 75
        else ("badge-red", "HIGH")
    )
)


# ============================================================
# CURRENT OPERATING STATE
# ============================================================

st.markdown(
    """
    <div class="section-header">
        Derived Operating State
        <div class="section-line"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        f"""
        <div class="card">
            <div class="metric-label">Thermal gradient</div>
            <div class="metric-value">
                {temperature_difference:.2f}
                <span style="font-size:0.8rem;color:#64748b;">K</span>
            </div>
            <div style="margin-top:7px;">
                <span class="badge {thermal_status[0]}">{thermal_status[1]}</span>
            </div>
            <div class="metric-description">Process temperature − ambient temperature</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        f"""
        <div class="card">
            <div class="metric-label">Power proxy</div>
            <div class="metric-value">{power_proxy:,.0f}</div>
            <div style="margin-top:7px;">
                <span class="badge badge-blue">DERIVED FEATURE</span>
            </div>
            <div class="metric-description">Rotational speed × spindle torque</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        f"""
        <div class="card">
            <div class="metric-label">Spindle load</div>
            <div class="metric-value">
                {torque:.1f}
                <span style="font-size:0.8rem;color:#64748b;">Nm</span>
            </div>
            <div style="margin-top:7px;">
                <span class="badge {torque_status[0]}">{torque_status[1]}</span>
            </div>
            <div class="metric-description">Current applied spindle torque</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m4:
    st.markdown(
        f"""
        <div class="card">
            <div class="metric-label">Tool wear</div>
            <div class="metric-value">
                {tool_wear}
                <span style="font-size:0.8rem;color:#64748b;">min</span>
            </div>
            <div style="margin-top:7px;">
                <span class="badge {wear_status[0]}">{wear_status[1]}</span>
            </div>
            <div class="metric-description">Accumulated operating time</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ML PIPELINE VISUALIZATION
# ============================================================

st.markdown(
    """
    <div class="section-header">
        Inference Pipeline
        <div class="section-line"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

p1, p2, p3, p4 = st.columns(4)

pipeline_items = [
    (p1, "01", "Sensor Inputs", "Temperature, RPM, torque and tool wear"),
    (p2, "02", "Feature Engineering", "Thermal gradient and power proxy"),
    (p3, "03", "ML Inference", "Serialized gradient boosting model"),
    (p4, "04", "Risk Decision", "Failure probability + threshold"),
]

for column, number, title, description in pipeline_items:
    with column:
        st.markdown(
            f"""
            <div class="pipeline-card">
                <div class="pipeline-number">STEP {number}</div>
                <div class="pipeline-title">{title}</div>
                <div class="pipeline-desc">{description}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# PREDICTION
# ============================================================

if predict_button:
    if process_temperature <= air_temperature:
        st.warning(
            "Process temperature is not above ambient temperature. "
            "Verify sensor calibration and telemetry quality before relying "
            "on the prediction."
        )

    # Standard internal schema
    input_dict = {
        "machine_type": machine_type,
        "air_temperature": float(air_temperature),
        "process_temperature": float(process_temperature),
        "rotational_speed": int(rotational_speed),
        "torque": float(torque),
        "tool_wear": int(tool_wear),
        "temperature_difference": float(temperature_difference),
        "power_proxy": float(power_proxy),
    }

    input_data = pd.DataFrame([input_dict])

    # Adapt schema automatically if pipeline was trained on original AI4I columns
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
        
        # Name translation mapping for common variations
        name_map = {
            "Type": machine_type,
            "Air temperature [K]": float(air_temperature),
            "Process temperature [K]": float(process_temperature),
            "Rotational speed [rpm]": int(rotational_speed),
            "Torque [Nm]": float(torque),
            "Tool wear [min]": int(tool_wear),
            "Temp_Diff": float(temperature_difference),
            "Power_Proxy": float(power_proxy),
        }

        # If model expects exact AI4I names, project them
        if any(col in name_map for col in expected_cols):
            mapped_data = {}
            for col in expected_cols:
                mapped_data[col] = name_map.get(col, input_dict.get(col, 0))
            inference_df = pd.DataFrame([mapped_data])[expected_cols]
        else:
            inference_df = input_data
    else:
        inference_df = input_data

    try:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(inference_df)[0]
            if hasattr(model, "classes_") and 1 in model.classes_:
                pos_idx = int(np.where(model.classes_ == 1)[0][0])
                probability = float(probabilities[pos_idx])
            else:
                probability = float(probabilities[-1])
        else:
            prediction = model.predict(inference_df)[0]
            probability = float(prediction)

    except Exception as error:
        st.error("Model inference failed.")
        st.exception(error)
        st.stop()

    probability = float(np.clip(probability, 0.0, 1.0))
    probability_percent = probability * 100.0
    threshold = 50.0
    is_failure = probability_percent >= threshold

    if probability_percent >= 50:
        risk_label = "HIGH RISK"
        risk_class = "prediction-danger"
        risk_color = "#ef4444"
        gauge_color = "#ef4444"
    elif probability_percent >= 30:
        risk_label = "WATCH"
        risk_class = "prediction-warning"
        risk_color = "#f59e0b"
        gauge_color = "#f59e0b"
    else:
        risk_label = "LOW RISK"
        risk_class = "prediction-success"
        risk_color = "#10b981"
        gauge_color = "#10b981"

    # ========================================================
    # RESULT
    # ========================================================

    st.markdown(
        """
        <div class="section-header">
            Diagnostic Result
            <div class="section-line"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    result_left, result_right = st.columns([1, 1.35])

    with result_left:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability_percent,
                number={
                    "suffix": "%",
                    "font": {
                        "family": "JetBrains Mono",
                        "size": 38,
                        "color": "#f8fafc",
                    },
                },
                title={
                    "text": "Failure probability",
                    "font": {
                        "family": "Inter",
                        "size": 13,
                        "color": "#94a3b8",
                    },
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 1,
                        "tickcolor": "#475569",
                        "tickfont": {"color": "#64748b"},
                    },
                    "bar": {
                        "color": gauge_color,
                        "thickness": 0.28,
                    },
                    "bgcolor": "rgba(255,255,255,0.025)",
                    "borderwidth": 1,
                    "bordercolor": "rgba(148,163,184,0.12)",
                    "steps": [
                        {"range": [0, 30], "color": "rgba(16,185,129,0.07)"},
                        {"range": [30, 50], "color": "rgba(245,158,11,0.07)"},
                        {"range": [50, 100], "color": "rgba(239,68,68,0.07)"},
                    ],
                    "threshold": {
                        "line": {"color": "#f8fafc", "width": 2},
                        "thickness": 0.8,
                        "value": threshold,
                    },
                },
            )
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=290,
            margin=dict(l=25, r=25, t=45, b=10),
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with result_right:
        st.html(
            f"""
            <div class="prediction-panel {risk_class}">
                <span class="badge"
                    style="
                        color:{risk_color};
                        background:{risk_color}12;
                        border:1px solid {risk_color}30;
                    ">
                    {risk_label}
                </span>

                <div class="prediction-title">
                    {"Elevated failure risk detected" if is_failure else "No immediate failure signal detected"}
                </div>

                <div class="prediction-copy">
                    The trained classification pipeline estimates a
                    <strong style="color:#f8fafc;">
                        {probability_percent:.2f}%
                    </strong>
                    probability of equipment failure under the
                    supplied operating conditions.
                    <br><br>
                    Decision threshold:
                    <span class="mono" style="color:#f8fafc;">
                        {threshold:.1f}%
                    </span>
                </div>

                <div class="action-box">
                    <div class="action-title">
                        Recommended interpretation
                    </div>
                    <div class="action-text">
                        {"Prioritize inspection of the cutting tool, thermal management system and spindle load. Use the prediction as a maintenance-triage signal rather than an autonomous shutdown command." if is_failure else "Continue normal monitoring while maintaining scheduled inspection intervals. The current telemetry does not cross the configured failure-risk threshold."}
                    </div>
                </div>
            </div>
            """
        )

    # ========================================================
    # FEATURE LEDGER
    # ========================================================

    st.markdown(
        """
        <div class="section-header">
            Model Input Ledger
            <div class="section-line"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_raw, tab_features, tab_schema = st.tabs(
        [
            "Raw Telemetry",
            "Engineered Features",
            "Inference Schema",
        ]
    )

    with tab_raw:
        raw_df = pd.DataFrame(
            {
                "Parameter": [
                    "Machine class",
                    "Air temperature",
                    "Process temperature",
                    "Rotational speed",
                    "Spindle torque",
                    "Tool wear",
                ],
                "Value": [
                    machine_type,
                    f"{air_temperature:.1f} K",
                    f"{process_temperature:.1f} K",
                    f"{rotational_speed:,} RPM",
                    f"{torque:.1f} Nm",
                    f"{tool_wear} min",
                ],
                "Type": [
                    "Categorical",
                    "Sensor",
                    "Sensor",
                    "Sensor",
                    "Sensor",
                    "Sensor",
                ],
            }
        )
        st.dataframe(raw_df, use_container_width=True, hide_index=True)

    with tab_features:
        engineered_df = pd.DataFrame(
            {
                "Feature": [
                    "temperature_difference",
                    "power_proxy",
                ],
                "Formula": [
                    "process_temperature − air_temperature",
                    "rotational_speed × torque",
                ],
                "Value": [
                    f"{temperature_difference:.2f} K",
                    f"{power_proxy:,.1f}",
                ],
                "Purpose": [
                    "Thermal stress indicator",
                    "Mechanical load proxy",
                ],
            }
        )
        st.dataframe(engineered_df, use_container_width=True, hide_index=True)

    with tab_schema:
        st.dataframe(input_data, use_container_width=True, hide_index=True)




# ========================================================
    # SHAP EXPLANATION
    # ========================================================

    st.markdown(
        """
        <div class="section-header">
            Model Explainability (SHAP)
            <div class="section-line"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="input-caption">
            SHAP (SHapley Additive exPlanations) breaks down how each sensor reading
            and engineered feature shifted the model's output away from baseline expectation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        # 1. Resolve pipeline steps and estimator
        if hasattr(model, "named_steps") and len(model.steps) > 1:
            step_names = list(model.named_steps.keys())
            final_model = model.named_steps[step_names[-1]]
            preprocessing_steps = Pipeline(model.steps[:-1])
        else:
            final_model = model
            preprocessing_steps = None

        # 2. Transform the aligned input dataframe
        target_input = inference_df.copy()
        if preprocessing_steps is not None:
            transformed_input = preprocessing_steps.transform(target_input)
        else:
            transformed_input = target_input

        if hasattr(transformed_input, "toarray"):
            transformed_input = transformed_input.toarray()

        # 3. Retrieve feature names
        try:
            if preprocessing_steps is not None and hasattr(preprocessing_steps, "get_feature_names_out"):
                feature_names = list(preprocessing_steps.get_feature_names_out())
            elif hasattr(model, "feature_names_in_"):
                feature_names = list(model.feature_names_in_)
            else:
                feature_names = [f"Feature {i + 1}" for i in range(transformed_input.shape[1])]
        except Exception:
            feature_names = [f"Feature {i + 1}" for i in range(transformed_input.shape[1])]

        # Simplify prefixed column names (e.g., "remainder__rotational_speed" -> "rotational_speed")
        clean_feature_names = [col.split("__")[-1] for col in feature_names]

        # Convert to DataFrame to ensure labels carry over to SHAP plots
        if not isinstance(transformed_input, pd.DataFrame):
            transformed_df = pd.DataFrame(transformed_input, columns=clean_feature_names)
        else:
            transformed_df = transformed_input.copy()
            transformed_df.columns = clean_feature_names

        # 4. Generate SHAP values (prefer TreeExplainer for tree/boosted models)
        try:
            explainer = shap.TreeExplainer(final_model)
            shap_obj = explainer(transformed_df)
        except Exception:
            explainer = shap.Explainer(final_model, transformed_df)
            shap_obj = explainer(transformed_df)

        # Handle binary classification slices (extract positive class failure risk)
        if len(shap_obj.shape) == 3 and shap_obj.shape[-1] == 2:
            shap_single = shap_obj[0, :, 1]
        else:
            shap_single = shap_obj[0]

        shap_left, shap_right = st.columns(2)

        # 5. SHAP Bar Chart
        with shap_left:
            st.markdown("##### Global Feature Contribution")
            fig_bar, ax_bar = plt.subplots(figsize=(7, 4.8))
            shap.plots.bar(shap_single, max_display=8, show=False)
            plt.tight_layout()
            st.pyplot(fig_bar, use_container_width=True)
            plt.close(fig_bar)

        # 6. SHAP Waterfall Chart
        with shap_right:
            st.markdown("##### Prediction Waterfall Breakdown")
            fig_waterfall = plt.figure(figsize=(7, 4.8))
            shap.plots.waterfall(shap_single, max_display=8, show=False)
            plt.tight_layout()
            st.pyplot(fig_waterfall, use_container_width=True)
            plt.close(fig_waterfall)

        st.caption(
            "Red bars push the diagnostic toward failure risk, while blue bars push "
            "it toward normal operating status."
        )

    except Exception as shap_error:
        st.warning("SHAP explanation could not be computed for this model configuration.")
        with st.expander("Diagnostic details"):
            st.exception(shap_error)

    # ========================================================
    # TECHNICAL SUMMARY
    # ========================================================

    st.markdown(
        """
        <div class="section-header">
            Technical Summary
            <div class="section-line"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    s1, s2, s3 = st.columns(3)

    with s1:
        st.markdown(
            """
            <div class="card">
                <div class="metric-label">Prediction task</div>
                <div style="color:#e2e8f0; font-size:0.95rem; font-weight:750; margin-top:8px;">
                    Binary classification
                </div>
                <div class="metric-description">
                    Estimates whether the current machine state
                    indicates elevated failure risk.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with s2:
        st.markdown(
            """
            <div class="card">
                <div class="metric-label">Feature strategy</div>
                <div style="color:#e2e8f0; font-size:0.95rem; font-weight:750; margin-top:8px;">
                    Sensor + engineered features
                </div>
                <div class="metric-description">
                    Combines direct telemetry with thermal and
                    mechanical load indicators.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with s3:
        st.markdown(
            """
            <div class="card">
                <div class="metric-label">Deployment</div>
                <div style="color:#e2e8f0; font-size:0.95rem; font-weight:750; margin-top:8px;">
                    Streamlit inference application
                </div>
                <div class="metric-description">
                    Serialized ML model loaded locally through
                    a reproducible inference pipeline.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <div>PREDIXAI · Predictive Maintenance</div>
        <div class="mono">AI4I 2020 · Streamlit · Plotly · Joblib</div>
    </div>
    """,
    unsafe_allow_html=True,
)