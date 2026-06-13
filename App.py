"""
NeuroScan AI - Professional Brain MRI Analysis
Developed by Mohammad Hasnain | Version 2.1
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
st.set_page_config(page_title="NeuroScan AI | Brain MRI", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

if 'scan_history' not in st.session_state:
    st.session_state.scan_history = []

# ── 2. CSS Styling ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #ffffff; margin-bottom: 0px;}
    .sub-header { font-size: 1.1rem; color: #94a3b8; margin-bottom: 20px; margin-top: 5px;}
    .result-card-healthy { background: linear-gradient(135deg, #064e3b, #022c22); border-left: 5px solid #10b981; padding: 20px; border-radius: 8px;}
    .result-card-tumor { background: linear-gradient(135deg, #7f1d1d, #450a0a); border-left: 5px solid #ef4444; padding: 20px; border-radius: 8px;}
    .disclaimer-box {
        background: linear-gradient(135deg, #1e1b4b, #0f0c29);
        border-left: 5px solid #818cf8;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 18px;
        color: #c7d2fe;
        font-size: 0.88rem;
        line-height: 1.6;
    }
    .disclaimer-box strong { color: #a5b4fc; }
    .disclaimer-title {
        font-size: 1rem;
        font-weight: 700;
        color: #a5b4fc;
        margin-bottom: 6px;
        letter-spacing: 0.03em;
    }
    .sidebar-disclaimer {
        background-color: #1e1b4b;
        border: 1px solid #4338ca;
        border-radius: 8px;
        padding: 12px;
        font-size: 0.78rem;
        color: #c7d2fe;
        line-height: 1.5;
        margin-top: 10px;
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
    st.markdown("<h2 style='text-align: center;'>🧠 NeuroScan AI</h2>", unsafe_allow_html=True)
    st.markdown("---")
    patient_name = st.text_input("Patient Name", "John Doe")
    patient_id = st.text_input("Patient ID", "MRN-84729")
    patient_age = st.number_input("Age", min_value=1, max_value=120, value=45)
    st.markdown("---")

    # ── Sidebar Disclaimer ──
    st.markdown("""
    <div class="sidebar-disclaimer">
        <strong>⚠️ Medical Disclaimer</strong><br><br>
        This tool is an AI-assisted screening aid only. It is <strong>not a certified medical device</strong> and must not be used as a substitute for professional medical diagnosis.<br><br>
        AI models can produce <strong>false positives or false negatives</strong>. All results <strong>must be reviewed and confirmed by a qualified radiologist or physician</strong> before any clinical decision is made.<br><br>
        <strong>Always consult a doctor.</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Developer:** Mohammad Hasnain")

# ── 5. Main UI ─────────────────────────────────────────────────
st.markdown('<p class="main-header">Automated Brain Tumor Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Professional Diagnostic Suite | Threshold: 65%</p>', unsafe_allow_html=True)

# ── Main Disclaimer Banner ──
st.markdown("""
<div class="disclaimer-box">
    <div class="disclaimer-title">⚠️ Important Medical Disclaimer — Please Read Before Use</div>
    NeuroScan AI is an <strong>experimental AI screening tool</strong> intended for research and educational purposes only.
    It is <strong>not approved</strong> for clinical or diagnostic use and should <strong>never replace</strong> the judgment of a licensed medical professional.<br><br>
    • AI models are not perfect and <strong>can make errors</strong>, including false positives (detecting a tumor that does not exist)
    and false negatives (missing a tumor that is present).<br>
    • Results produced by this system are <strong>not a medical diagnosis</strong>.<br>
    • <strong>Always consult a qualified radiologist or neurologist</strong> to interpret MRI scans and confirm any findings.<br><br>
    <strong>If you or a patient may have a medical emergency, contact emergency services immediately.</strong>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔬 Diagnostic Scanner", "🕰️ Session History"])

with tab1:
    model, model_loaded = load_vision_model()
    if not model_loaded:
        st.error("⚠️ Critical System Error: Unable to load neural network.")
        st.stop()

    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.markdown("### 📤 Upload MRI Scan")
        uploaded_file = st.file_uploader("Select MRI Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded_file is not None:
            st.image(uploaded_file, caption='Patient MRI Scan', use_container_width=True)

    with col2:
        st.markdown("### 🔬 Diagnostic Report")
        if uploaded_file is None:
            st.info("Upload an MRI scan on the left to begin.")
        else:
            if st.button("▶ INITIATE AI ANALYSIS", type="primary", use_container_width=True):
                # Progress bar simulation
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                progress_bar.empty()

                # Predict
                image = Image.open(uploaded_file).convert('RGB')
                img_resized = image.resize((150, 150))
                img_array = np.expand_dims(np.array(img_resized), axis=0)

                prob = model.predict(img_array, verbose=0)[0][0]

                # CLINICAL THRESHOLD: 65%
                is_tumor = prob >= 0.65
                conf_percent = (prob * 100) if is_tumor else ((1 - prob) * 100)

                if is_tumor:
                    st.markdown(f"""<div class="result-card-tumor">
                        <h2 style="color: #fca5a5; margin-top: 0;">⚠️ ABNORMALITY DETECTED</h2>
                        <h1 style="color: white; margin-bottom: 0;">{conf_percent:.2f}%</h1>
                        <p style="color: #fca5a5; margin-top: 0; font-size: 0.9rem;">CONFIDENCE | RISK: HIGH</p>
                    </div>""", unsafe_allow_html=True)
                    result_text = "TUMOR DETECTED"
                    # Post-result disclaimer for positive findings
                    st.warning("⚠️ This result indicates a possible abnormality. **This is not a confirmed diagnosis.** Please seek immediate consultation with a qualified neurologist or radiologist for further evaluation.")
                else:
                    st.markdown(f"""<div class="result-card-healthy">
                        <h2 style="color: #6ee7b7; margin-top: 0;">✅ SCAN APPEARS NORMAL</h2>
                        <h1 style="color: white; margin-bottom: 0;">{conf_percent:.2f}%</h1>
                        <p style="color: #6ee7b7; margin-top: 0; font-size: 0.9rem;">CONFIDENCE | RISK: LOW</p>
                    </div>""", unsafe_allow_html=True)
                    result_text = "HEALTHY (NO TUMOR)"
                    # Post-result disclaimer for negative findings
                    st.info("ℹ️ A normal result does not guarantee the complete absence of pathology. Always confirm with a certified medical professional.")

                # Save History
                scan_record = {
                    "Date": datetime.datetime.now().strftime("%H:%M:%S"),
                    "Patient": patient_name,
                    "ID": patient_id,
                    "Age": patient_age,
                    "Diagnosis": result_text,
                    "Confidence": f"{conf_percent:.2f}%"
                }
                st.session_state.scan_history.append(scan_record)

                report_content = f"""NeuroScan AI — Diagnostic Report
================================
Generated : {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Patient   : {patient_name}
ID        : {patient_id}
Age       : {patient_age}

AI Result : {result_text}
Confidence: {conf_percent:.2f}%

DISCLAIMER
----------
This report is generated by an AI model and is for screening
purposes only. It does not constitute a medical diagnosis.
All findings must be verified by a licensed radiologist or
neurologist before any clinical decision is made.
"""
                st.download_button(
                    "📄 DOWNLOAD REPORT",
                    data=report_content,
                    file_name=f"NeuroScan_Report_{patient_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    use_container_width=True
                )

with tab2:
    if st.session_state.scan_history:
        st.table(st.session_state.scan_history)
    else:
        st.info("No scans performed in this session yet.")
