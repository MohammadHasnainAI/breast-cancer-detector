"""
NeuroScan AI - 3D Professional Diagnostic Suite
Developed by Mohammad Hasnain | Version 3.1
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
st.set_page_config(page_title="NeuroScan AI | 3D Dashboard", page_icon="🧠", layout="wide")

# Persistent History Tracker
if 'scan_history' not in st.session_state:
    st.session_state.scan_history = []

# ── 2. 3D CSS Animation Background ─────────────────────────────
st.markdown("""
<style>
    .stApp { background: radial-gradient(circle at 50% 50%, #0f172a 0%, #020617 100%); }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
    }
    .main-header { font-size: 2.5rem; font-weight: 800; color: #ffffff; text-align: center; margin-bottom: 30px; }
</style>
""", unsafe_allow_html=True)

# ── 3. Model Loading ───────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_vision_model():
    try:
        if not os.path.exists('brain_tumor_model.h5'):
            file_id = '13qow6nsxNW0SsiQBPe2QPVnpB3NGB7FN'
            gdown.download(f'https://drive.google.com/uc?id={file_id}', 'brain_tumor_model.h5', quiet=False)
        return tf.keras.models.load_model('brain_tumor_model.h5'), True
    except: return None, False

model, model_loaded = load_vision_model()

# ── 4. Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 NeuroScan AI Panel")
    patient_name = st.text_input("Patient Name", "John Doe")
    patient_id = st.text_input("Patient ID", "MRN-84729")
    st.markdown("---")
    st.info("System Ready: 3D Visualization Active")

# ── 5. Main Dashboard ──────────────────────────────────────────
st.markdown('<p class="main-header">🧠 NeuroScan 3D Analysis</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔬 Diagnostic Suite", "🕰️ Patient History"])

with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="glass-card"><h3>📤 Upload MRI Scan</h3>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Select Scan", type=["jpg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card"><h3>🔬 AI Diagnostic Engine</h3>', unsafe_allow_html=True)
        if uploaded_file and st.button("▶ EXECUTE 3D SCAN", type="primary"):
            img_resized = image.resize((150, 150))
            img_array = np.expand_dims(np.array(img_resized), axis=0)
            prob = model.predict(img_array, verbose=0)[0][0]
            
            with st.spinner("Processing 3D Neural Topology..."):
                time.sleep(1.5)
            
            # Prediction logic
            diagnosis = "TUMOR DETECTED" if prob >= 0.5 else "HEALTHY"
            color = "#ef4444" if prob >= 0.5 else "#10b981"
            
            st.markdown(f"<h2 style='color:{color}'>{diagnosis}</h2>", unsafe_allow_html=True)
            st.metric("AI Confidence", f"{prob*100:.2f}%" if prob >= 0.5 else f"{(1-prob)*100:.2f}%")
            
            # Save History
            st.session_state.scan_history.append({"Date": datetime.datetime.now().strftime("%H:%M"), "Patient": patient_name, "Result": diagnosis})
            
            # Download Report
            report = f"Report for {patient_name}: {diagnosis}"
            st.download_button("📄 Download Report", report, file_name="report.txt")
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("### 🕰️ Session History")
    st.table(st.session_state.scan_history)
