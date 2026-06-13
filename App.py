"""
NeuroScan AI - Professional Brain MRI Analysis
Developed by Mohammad Hasnain | Version 3.0
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
import time
import datetime

# ── 1. Page Configuration ──────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI | Brain MRI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'scan_history' not in st.session_state:
    st.session_state.scan_history = []

# ── 2. CSS Styling ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Page Background ── */
    .stApp {
        background-color: #0d1117;
    }

    /* ── Header ── */
    .ns-header {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 28px 0 6px 0;
        border-bottom: 1px solid #1e2a3a;
        margin-bottom: 24px;
    }
    .ns-logo {
        font-size: 2.2rem;
        line-height: 1;
    }
    .ns-title {
        font-size: 1.7rem;
        font-weight: 700;
        color: #e2e8f0;
        letter-spacing: -0.01em;
        margin: 0;
    }
    .ns-subtitle {
        font-size: 0.82rem;
        color: #4a7fa5;
        margin: 2px 0 0 0;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-weight: 500;
    }

    /* ── Disclaimer Banner ── */
    .ns-disclaimer {
        background-color: #111827;
        border: 1px solid #1e3a5f;
        border-left: 4px solid #3b82f6;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 22px;
        color: #94a3b8;
        font-size: 0.82rem;
        line-height: 1.6;
    }
    .ns-disclaimer strong { color: #60a5fa; }

    /* ── Upload Zone ── */
    .upload-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #4a7fa5;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 8px;
        display: block;
    }

    /* ── Section Heading ── */
    .ns-section-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #4a7fa5;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 14px;
        padding-bottom: 8px;
        border-bottom: 1px solid #1e2a3a;
    }

    /* ── Result Cards ── */
    .result-positive {
        background: #160a0a;
        border: 1px solid #7f1d1d;
        border-left: 5px solid #ef4444;
        border-radius: 8px;
        padding: 20px 22px;
        margin-bottom: 14px;
    }
    .result-positive .r-label { color: #f87171; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 4px; }
    .result-positive .r-value { color: #ffffff; font-size: 2.4rem; font-weight: 700; line-height: 1; margin-bottom: 2px; }
    .result-positive .r-sub   { color: #fca5a5; font-size: 0.78rem; margin-top: 4px; }

    .result-negative {
        background: #030f0a;
        border: 1px solid #064e3b;
        border-left: 5px solid #10b981;
        border-radius: 8px;
        padding: 20px 22px;
        margin-bottom: 14px;
    }
    .result-negative .r-label { color: #34d399; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 4px; }
    .result-negative .r-value { color: #ffffff; font-size: 2.4rem; font-weight: 700; line-height: 1; margin-bottom: 2px; }
    .result-negative .r-sub   { color: #6ee7b7; font-size: 0.78rem; margin-top: 4px; }

    /* ── Inline Notice ── */
    .notice-red {
        background: #1a0808;
        border: 1px solid #7f1d1d;
        border-radius: 6px;
        padding: 10px 14px;
        color: #fca5a5;
        font-size: 0.81rem;
        margin-top: 10px;
        line-height: 1.5;
    }
    .notice-blue {
        background: #050d1a;
        border: 1px solid #1e3a5f;
        border-radius: 6px;
        padding: 10px 14px;
        color: #93c5fd;
        font-size: 0.81rem;
        margin-top: 10px;
        line-height: 1.5;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: #0b0f19;
        border-right: 1px solid #1e2a3a;
    }
    .sidebar-brand {
        text-align: center;
        padding: 10px 0 6px 0;
    }
    .sidebar-brand .s-icon { font-size: 2rem; }
    .sidebar-brand .s-name { font-size: 1rem; font-weight: 700; color: #e2e8f0; margin-top: 4px; }
    .sidebar-brand .s-ver  { font-size: 0.7rem; color: #4a7fa5; letter-spacing: 0.08em; text-transform: uppercase; }

    .sidebar-section {
        font-size: 0.68rem;
        font-weight: 700;
        color: #4a7fa5;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin: 18px 0 8px 0;
    }

    .sidebar-disclaimer {
        background: #111827;
        border: 1px solid #1e3a5f;
        border-radius: 6px;
        padding: 10px 12px;
        color: #64748b;
        font-size: 0.72rem;
        line-height: 1.55;
        margin-top: 6px;
    }
    .sidebar-disclaimer strong { color: #3b82f6; }

    .sidebar-dev {
        text-align: center;
        font-size: 0.7rem;
        color: #334155;
        margin-top: 18px;
    }

    /* ── History Table ── */
    .stDataFrame { border: 1px solid #1e2a3a !important; border-radius: 6px; overflow: hidden; }

    /* ── Buttons ── */
    .stButton > button {
        background: #1e40af;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.83rem;
        letter-spacing: 0.04em;
        padding: 10px;
        transition: background 0.2s;
    }
    .stButton > button:hover { background: #1d4ed8; }

    .stDownloadButton > button {
        background: #0f172a;
        color: #60a5fa;
        border: 1px solid #1e3a5f;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }

    /* ── Divider ── */
    hr { border-color: #1e2a3a !important; }

    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #0b0f19;
        border-bottom: 1px solid #1e2a3a;
        padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        color: #4a7fa5;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        padding: 8px 16px;
        border-radius: 4px 4px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background: #0d1117;
        color: #e2e8f0 !important;
        border-bottom: 2px solid #3b82f6;
    }

    /* ── Input fields ── */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background: #111827;
        border: 1px solid #1e2a3a;
        border-radius: 6px;
        color: #e2e8f0;
        font-size: 0.85rem;
    }
    label[data-testid="stWidgetLabel"] p {
        color: #64748b;
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# ── 3. Model Loading ───────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_vision_model():
    try:
        if not os.path.exists('brain_tumor_model.h5'):
            file_id = '13qow6nsxNW0SsiQBPe2QPVnpB3NGB7FN'
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, 'brain_tumor_model.h5', quiet=False)
        model = tf.keras.models.load_model('brain_tumor_model.h5')
        return model, True
    except Exception as e:
        return None, False

# ── 4. Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="s-icon">🧠</div>
        <div class="s-name">NeuroScan AI</div>
        <div class="s-ver">Version 3.0 · Brain MRI</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-section">Patient Info</div>', unsafe_allow_html=True)

    patient_name = st.text_input("Patient Name", "John Doe")
    patient_id   = st.text_input("Patient ID",   "MRN-84729")
    patient_age  = st.number_input("Age", min_value=1, max_value=120, value=45)

    st.markdown("---")
    st.markdown('<div class="sidebar-section">Notice</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-disclaimer">
        <strong>⚠️ For clinical use only with doctor review.</strong><br><br>
        This AI tool can make mistakes. It does not replace a qualified doctor's diagnosis. Always confirm results with a radiologist.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-dev">Developed by Mohammad Hasnain</div>', unsafe_allow_html=True)

# ── 5. Header ─────────────────────────────────────────────────
st.markdown("""
<div class="ns-header">
    <div class="ns-logo">🧠</div>
    <div>
        <p class="ns-title">NeuroScan AI</p>
        <p class="ns-subtitle">Brain MRI · Tumor Detection · AI-Assisted Screening</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Disclaimer Banner ──
st.markdown("""
<div class="ns-disclaimer">
    <strong>⚠️ Medical Notice:</strong>&nbsp; This tool helps doctors screen MRI scans faster — it does not replace their judgment.
    The AI can be wrong. <strong>All results must be confirmed by a licensed doctor before any action is taken.</strong>
</div>
""", unsafe_allow_html=True)

# ── 6. Tabs ────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["  🔬  Scan  ", "  🕰️  History  "])

with tab1:
    with st.spinner("Loading AI model..."):
        model, model_loaded = load_vision_model()

    if not model_loaded:
        st.error("Could not load the AI model. Please refresh the page or contact support.")
        st.stop()

    col1, col2 = st.columns([1, 1.2], gap="large")

    # ── Left: Upload ──
    with col1:
        st.markdown('<div class="ns-section-title">Upload MRI Scan</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop an MRI image here (JPG or PNG)",
            type=["jpg", "jpeg", "png"],
            help="Upload a brain MRI image. Supported formats: JPG, PNG."
        )
        if uploaded_file:
            st.image(uploaded_file, caption=f"Patient: {patient_name}", use_container_width=True)

    # ── Right: Report ──
    with col2:
        st.markdown('<div class="ns-section-title">Diagnostic Report</div>', unsafe_allow_html=True)

        if not uploaded_file:
            st.markdown("""
            <div style="background:#0b0f19; border:1px dashed #1e2a3a; border-radius:8px; padding:32px 20px; text-align:center; color:#334155;">
                <div style="font-size:2rem; margin-bottom:10px;">📂</div>
                <div style="font-size:0.85rem; font-weight:500; color:#4a7fa5;">Upload an MRI scan on the left to begin analysis</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            run = st.button("▶  Run AI Analysis", type="primary", use_container_width=True)

            if run:
                with st.spinner("Analysing scan..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.008)
                        progress_bar.progress(i + 1)
                    progress_bar.empty()

                    image       = Image.open(uploaded_file).convert('RGB')
                    img_resized = image.resize((150, 150))
                    img_array   = np.expand_dims(np.array(img_resized), axis=0)
                    prob        = model.predict(img_array, verbose=0)[0][0]

                is_tumor    = prob >= 0.65
                conf_pct    = (prob * 100) if is_tumor else ((1 - prob) * 100)
                timestamp   = datetime.datetime.now().strftime("%d %b %Y · %H:%M")

                # Metadata row
                st.markdown(f"""
                <div style="display:flex; gap:20px; margin-bottom:14px;">
                    <div style="background:#0b0f19; border:1px solid #1e2a3a; border-radius:6px; padding:10px 16px; flex:1;">
                        <div style="font-size:0.65rem; color:#4a7fa5; text-transform:uppercase; letter-spacing:0.1em; font-weight:600;">Patient</div>
                        <div style="color:#e2e8f0; font-size:0.88rem; font-weight:500; margin-top:3px;">{patient_name}</div>
                    </div>
                    <div style="background:#0b0f19; border:1px solid #1e2a3a; border-radius:6px; padding:10px 16px; flex:1;">
                        <div style="font-size:0.65rem; color:#4a7fa5; text-transform:uppercase; letter-spacing:0.1em; font-weight:600;">ID / Age</div>
                        <div style="color:#e2e8f0; font-size:0.88rem; font-weight:500; margin-top:3px;">{patient_id} · {patient_age} yrs</div>
                    </div>
                    <div style="background:#0b0f19; border:1px solid #1e2a3a; border-radius:6px; padding:10px 16px; flex:1;">
                        <div style="font-size:0.65rem; color:#4a7fa5; text-transform:uppercase; letter-spacing:0.1em; font-weight:600;">Scanned</div>
                        <div style="color:#e2e8f0; font-size:0.88rem; font-weight:500; margin-top:3px;">{timestamp}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Result card
                if is_tumor:
                    st.markdown(f"""
                    <div class="result-positive">
                        <div class="r-label">⚠️ Abnormality Detected</div>
                        <div class="r-value">{conf_pct:.1f}%</div>
                        <div class="r-sub">AI Confidence · Risk Level: HIGH — Radiologist review required</div>
                    </div>
                    <div class="notice-red">
                        The scan shows signs that may indicate a tumor. Please refer this patient to a neurologist or radiologist for confirmation.
                    </div>
                    """, unsafe_allow_html=True)
                    result_text = "Abnormality Detected"
                else:
                    st.markdown(f"""
                    <div class="result-negative">
                        <div class="r-label">✅ No Abnormality Found</div>
                        <div class="r-value">{conf_pct:.1f}%</div>
                        <div class="r-sub">AI Confidence · Risk Level: LOW — Routine follow-up advised</div>
                    </div>
                    <div class="notice-blue">
                        The scan appears normal. A normal AI result does not rule out all conditions. Use clinical judgment alongside this result.
                    </div>
                    """, unsafe_allow_html=True)
                    result_text = "No Abnormality Found"

                # Save to history
                st.session_state.scan_history.append({
                    "Time"     : datetime.datetime.now().strftime("%H:%M:%S"),
                    "Patient"  : patient_name,
                    "ID"       : patient_id,
                    "Age"      : patient_age,
                    "Result"   : result_text,
                    "Confidence": f"{conf_pct:.1f}%"
                })

                # Report download
                st.markdown("<br>", unsafe_allow_html=True)
                report_txt = f"""NEUROSCAN AI — DIAGNOSTIC REPORT
================================
Date      : {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Patient   : {patient_name}
ID        : {patient_id}
Age       : {patient_age} years

AI Result : {result_text}
Confidence: {conf_pct:.1f}%

MEDICAL DISCLAIMER
------------------
This report is generated by an AI screening tool and is NOT
a medical diagnosis. All findings must be reviewed and confirmed
by a licensed radiologist or neurologist before any clinical
decision is made. The AI model can make errors.

Developed by Mohammad Hasnain | NeuroScan AI v3.0
"""
                st.download_button(
                    "📄  Download Report (TXT)",
                    data=report_txt,
                    file_name=f"NeuroScan_{patient_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    use_container_width=True
                )

# ── History Tab ───────────────────────────────────────────────
with tab2:
    st.markdown('<div class="ns-section-title">Session Scan History</div>', unsafe_allow_html=True)
    if st.session_state.scan_history:
        st.dataframe(
            st.session_state.scan_history,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.markdown("""
        <div style="background:#0b0f19; border:1px dashed #1e2a3a; border-radius:8px; padding:32px; text-align:center; color:#334155;">
            <div style="font-size:1.6rem; margin-bottom:8px;">🕰️</div>
            <div style="font-size:0.85rem; color:#4a7fa5;">No scans in this session yet.</div>
        </div>
        """, unsafe_allow_html=True)
