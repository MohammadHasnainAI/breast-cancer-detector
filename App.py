"""
NeuroScan AI - Professional Brain MRI Analysis
Developed by Mohammad Hasnain | Version 4.0
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
import time
import datetime

# ── 1. Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI | Brain MRI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 2. Session State ───────────────────────────────────────────
if 'scan_history' not in st.session_state:
    st.session_state.scan_history = []
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'last_result' not in st.session_state:
    st.session_state.last_result = None

# ── 3. Theme Colors ────────────────────────────────────────────
if st.session_state.dark_mode:
    BG        = "#1a1a2e"
    BG2       = "#16213e"
    BG3       = "#0f3460"
    BORDER    = "#2d3748"
    TEXT      = "#e2e8f0"
    TEXT2     = "#94a3b8"
    ACCENT    = "#4299e1"
    CARD_BG   = "#1e2a3a"
    INPUT_BG  = "#1e2a3a"
    SHADOW    = "rgba(0,0,0,0.3)"
else:
    BG        = "#f1f5f9"
    BG2       = "#ffffff"
    BG3       = "#e2e8f0"
    BORDER    = "#cbd5e1"
    TEXT      = "#1e293b"
    TEXT2     = "#64748b"
    ACCENT    = "#2563eb"
    CARD_BG   = "#ffffff"
    INPUT_BG  = "#ffffff"
    SHADOW    = "rgba(0,0,0,0.08)"

# ── 4. CSS ─────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

/* Page background */
.stApp {{
    background-color: {BG};
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background-color: {BG2};
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stNumberInput input {{
    background-color: {INPUT_BG};
    border: 1px solid {BORDER};
    color: {TEXT};
    border-radius: 6px;
}}

/* Labels */
label[data-testid="stWidgetLabel"] p {{
    color: {TEXT2};
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background-color: {BG};
    border-bottom: 2px solid {BORDER};
    gap: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    color: {TEXT2};
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 8px 18px;
    border-radius: 6px 6px 0 0;
    background: transparent;
}}
.stTabs [aria-selected="true"] {{
    color: {ACCENT} !important;
    background-color: {BG2} !important;
    border-bottom: 2px solid {ACCENT};
}}

/* Buttons */
.stButton > button {{
    background: {ACCENT};
    color: white;
    border: none;
    border-radius: 7px;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.03em;
    padding: 11px;
    width: 100%;
    transition: opacity 0.2s;
    cursor: pointer;
}}
.stButton > button:hover {{
    opacity: 0.88;
}}

.stDownloadButton > button {{
    background: {BG2};
    color: {ACCENT};
    border: 1.5px solid {ACCENT};
    border-radius: 7px;
    font-weight: 600;
    font-size: 0.82rem;
}}

/* File uploader */
[data-testid="stFileUploaderDropzone"] {{
    background: {BG2};
    border: 2px dashed {BORDER};
    border-radius: 10px;
}}

/* Dataframe */
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    overflow: hidden;
}}

/* Progress bar */
.stProgress > div > div > div {{
    background: {ACCENT};
    border-radius: 4px;
}}

/* General text */
p, div, span, h1, h2, h3 {{
    color: {TEXT};
}}
</style>
""", unsafe_allow_html=True)

# ── 5. Model Loading ───────────────────────────────────────────
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

# ── 6. Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown(f"""
    <div style="text-align:center; padding:16px 0 10px 0;">
        <div style="font-size:2.2rem;">🧠</div>
        <div style="font-size:1.05rem; font-weight:700; color:{TEXT}; margin-top:4px;">NeuroScan AI</div>
        <div style="font-size:0.68rem; color:{TEXT2}; letter-spacing:0.1em; text-transform:uppercase; margin-top:2px;">Brain MRI · Version 4.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Dark mode toggle
    mode_label = "☀️ Light Mode" if st.session_state.dark_mode else "🌙 Dark Mode"
    if st.button(mode_label, use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.session_state.analysis_done = False
        st.rerun()

    st.divider()

    # Patient info
    st.markdown(f"<div style='font-size:0.7rem; font-weight:700; color:{TEXT2}; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:10px;'>Patient Information</div>", unsafe_allow_html=True)
    patient_name = st.text_input("Patient Name", "John Doe")
    patient_id   = st.text_input("Patient ID",   "MRN-84729")
    patient_age  = st.number_input("Age", min_value=1, max_value=120, value=45)

    st.divider()

    # Disclaimer
    st.markdown(f"""
    <div style="background:{BG3}; border-left:3px solid {ACCENT}; border-radius:6px; padding:11px 13px; font-size:0.75rem; color:{TEXT2}; line-height:1.6;">
        <strong style="color:{ACCENT};">⚠️ Medical Notice</strong><br><br>
        This AI tool is for screening only. It <strong>can make mistakes</strong> and does not replace a doctor's diagnosis.<br><br>
        <strong>Always confirm results with a qualified radiologist.</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div style='text-align:center; font-size:0.68rem; color:{TEXT2}; margin-top:20px;'>Developed by Mohammad Hasnain</div>", unsafe_allow_html=True)

# ── 7. Header ──────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex; align-items:center; gap:14px; padding:20px 0 10px 0; border-bottom:1px solid {BORDER}; margin-bottom:20px;">
    <div style="font-size:2rem;">🧠</div>
    <div>
        <div style="font-size:1.6rem; font-weight:700; color:{TEXT}; letter-spacing:-0.01em; line-height:1.2;">NeuroScan AI</div>
        <div style="font-size:0.75rem; color:{TEXT2}; letter-spacing:0.08em; text-transform:uppercase; margin-top:3px;">AI-Assisted Brain Tumor Screening</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Disclaimer Banner ──
st.markdown(f"""
<div style="background:{BG2}; border:1px solid {BORDER}; border-left:4px solid {ACCENT}; border-radius:7px; padding:11px 16px; margin-bottom:20px; font-size:0.81rem; color:{TEXT2}; line-height:1.6; box-shadow:0 1px 4px {SHADOW};">
    <strong style="color:{ACCENT};">⚠️ For clinical use alongside a doctor.</strong>
    &nbsp;This AI helps scan MRI images faster — it is not a diagnosis tool. Results must always be reviewed by a licensed doctor before any medical decision.
</div>
""", unsafe_allow_html=True)

# ── 8. Tabs ────────────────────────────────────────────────────
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
        st.markdown(f"<div style='font-size:0.72rem; font-weight:700; color:{TEXT2}; letter-spacing:0.1em; text-transform:uppercase; padding-bottom:8px; border-bottom:1px solid {BORDER}; margin-bottom:14px;'>Upload MRI Scan</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop an MRI image here (JPG or PNG)",
            type=["jpg", "jpeg", "png"],
            help="Upload a brain MRI image in JPG or PNG format."
        )
        if uploaded_file:
            # Reset analysis when a new file is uploaded
            if st.session_state.get('last_file') != uploaded_file.name:
                st.session_state.analysis_done = False
                st.session_state.last_result   = None
                st.session_state['last_file']  = uploaded_file.name
            st.image(uploaded_file, caption=f"Patient: {patient_name}", use_container_width=True)

    # ── Right: Report ──
    with col2:
        st.markdown(f"<div style='font-size:0.72rem; font-weight:700; color:{TEXT2}; letter-spacing:0.1em; text-transform:uppercase; padding-bottom:8px; border-bottom:1px solid {BORDER}; margin-bottom:14px;'>Diagnostic Report</div>", unsafe_allow_html=True)

        if not uploaded_file:
            st.markdown(f"""
            <div style="background:{BG2}; border:2px dashed {BORDER}; border-radius:10px; padding:40px 20px; text-align:center; box-shadow:0 1px 4px {SHADOW};">
                <div style="font-size:2rem; margin-bottom:10px;">📂</div>
                <div style="font-size:0.85rem; font-weight:500; color:{TEXT2};">Upload an MRI scan on the left to get started</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            if st.button("▶  Run AI Analysis", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.007)
                    progress_bar.progress(i + 1)
                progress_bar.empty()

                image       = Image.open(uploaded_file).convert('RGB')
                img_resized = image.resize((150, 150))
                img_array   = np.expand_dims(np.array(img_resized), axis=0)
                prob        = float(model.predict(img_array, verbose=0)[0][0])
                is_tumor    = prob >= 0.65
                conf_pct    = (prob * 100) if is_tumor else ((1 - prob) * 100)
                timestamp   = datetime.datetime.now().strftime("%d %b %Y · %H:%M")
                result_text = "Abnormality Detected" if is_tumor else "No Abnormality Found"

                st.session_state.analysis_done = True
                st.session_state.last_result   = {
                    "is_tumor"   : is_tumor,
                    "conf_pct"   : conf_pct,
                    "timestamp"  : timestamp,
                    "result_text": result_text,
                }
                st.session_state.scan_history.append({
                    "Time"      : datetime.datetime.now().strftime("%H:%M:%S"),
                    "Patient"   : patient_name,
                    "ID"        : patient_id,
                    "Age"       : patient_age,
                    "Result"    : result_text,
                    "Confidence": f"{conf_pct:.1f}%"
                })

            # Show result if available
            if st.session_state.analysis_done and st.session_state.last_result:
                r         = st.session_state.last_result
                is_tumor  = r["is_tumor"]
                conf_pct  = r["conf_pct"]
                timestamp = r["timestamp"]
                result_text = r["result_text"]

                # Metadata row
                st.markdown(f"""
                <div style="display:flex; gap:10px; margin-bottom:14px; margin-top:6px;">
                    <div style="background:{BG3}; border:1px solid {BORDER}; border-radius:7px; padding:10px 14px; flex:1;">
                        <div style="font-size:0.62rem; color:{TEXT2}; text-transform:uppercase; letter-spacing:0.1em; font-weight:600;">Patient</div>
                        <div style="color:{TEXT}; font-size:0.85rem; font-weight:600; margin-top:3px;">{patient_name}</div>
                    </div>
                    <div style="background:{BG3}; border:1px solid {BORDER}; border-radius:7px; padding:10px 14px; flex:1;">
                        <div style="font-size:0.62rem; color:{TEXT2}; text-transform:uppercase; letter-spacing:0.1em; font-weight:600;">ID / Age</div>
                        <div style="color:{TEXT}; font-size:0.85rem; font-weight:600; margin-top:3px;">{patient_id} · {patient_age} yrs</div>
                    </div>
                    <div style="background:{BG3}; border:1px solid {BORDER}; border-radius:7px; padding:10px 14px; flex:1;">
                        <div style="font-size:0.62rem; color:{TEXT2}; text-transform:uppercase; letter-spacing:0.1em; font-weight:600;">Scanned</div>
                        <div style="color:{TEXT}; font-size:0.85rem; font-weight:600; margin-top:3px;">{timestamp}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if is_tumor:
                    st.markdown(f"""
                    <div style="background:#fff5f5; border:1px solid #fc8181; border-left:5px solid #e53e3e; border-radius:8px; padding:18px 20px; margin-bottom:10px;">
                        <div style="font-size:0.7rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:#e53e3e; margin-bottom:4px;">⚠️ Abnormality Detected</div>
                        <div style="font-size:2.4rem; font-weight:700; color:#1a202c; line-height:1;">{conf_pct:.1f}%</div>
                        <div style="font-size:0.78rem; color:#c53030; margin-top:5px;">AI Confidence · Risk Level: HIGH</div>
                    </div>
                    <div style="background:#fff5f5; border:1px solid #fc8181; border-radius:6px; padding:10px 14px; font-size:0.81rem; color:#c53030; line-height:1.55; margin-bottom:14px;">
                        The scan shows signs that may indicate a tumor. Please refer this patient to a neurologist or radiologist for confirmation.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background:#f0fff4; border:1px solid #68d391; border-left:5px solid #38a169; border-radius:8px; padding:18px 20px; margin-bottom:10px;">
                        <div style="font-size:0.7rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:#38a169; margin-bottom:4px;">✅ No Abnormality Found</div>
                        <div style="font-size:2.4rem; font-weight:700; color:#1a202c; line-height:1;">{conf_pct:.1f}%</div>
                        <div style="font-size:0.78rem; color:#276749; margin-top:5px;">AI Confidence · Risk Level: LOW</div>
                    </div>
                    <div style="background:#ebf8ff; border:1px solid #63b3ed; border-radius:6px; padding:10px 14px; font-size:0.81rem; color:#2b6cb0; line-height:1.55; margin-bottom:14px;">
                        The scan appears normal. A normal AI result does not rule out all conditions — always use clinical judgment alongside this result.
                    </div>
                    """, unsafe_allow_html=True)

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
This report is produced by an AI screening tool.
It is NOT a medical diagnosis. All findings must be
confirmed by a licensed radiologist or neurologist
before any clinical decision is made.

Developed by Mohammad Hasnain | NeuroScan AI v4.0
"""
                st.download_button(
                    "📄  Download Report",
                    data=report_txt,
                    file_name=f"NeuroScan_{patient_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    use_container_width=True
                )

# ── History Tab ───────────────────────────────────────────────
with tab2:
    st.markdown(f"<div style='font-size:0.72rem; font-weight:700; color:{TEXT2}; letter-spacing:0.1em; text-transform:uppercase; padding-bottom:8px; border-bottom:1px solid {BORDER}; margin-bottom:14px;'>Session Scan History</div>", unsafe_allow_html=True)
    if st.session_state.scan_history:
        st.dataframe(st.session_state.scan_history, use_container_width=True, hide_index=True)
    else:
        st.markdown(f"""
        <div style="background:{BG2}; border:2px dashed {BORDER}; border-radius:10px; padding:36px; text-align:center;">
            <div style="font-size:1.6rem; margin-bottom:8px;">🕰️</div>
            <div style="font-size:0.85rem; color:{TEXT2};">No scans yet in this session.</div>
        </div>
        """, unsafe_allow_html=True)
