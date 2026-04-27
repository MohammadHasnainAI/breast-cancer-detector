"""
NeuroScan AI - Brain Tumor MRI Analysis Dashboard
Developed by Mohammad Hasnain | Version 2.0
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance
import gdown
import os
import time
import datetime

# ── 1. Page Configuration ──────────────────────────────────────
st.set_page_config(page_title="NeuroScan AI | Brain MRI", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# Initialize Session State for History (Function 5)
if 'scan_history' not in st.session_state:
    st.session_state.scan_history = []

# ── 2. Custom Professional CSS ─────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-header { font-size: 2.5rem; font-weight: 700; color: #ffffff; margin-bottom: 0px;}
    .sub-header { font-size: 1.1rem; color: #94a3b8; margin-bottom: 20px; margin-top: 5px;}
    .result-card-healthy { background: linear-gradient(135deg, #064e3b, #022c22); border-left: 5px solid #10b981; padding: 20px; border-radius: 8px;}
    .result-card-tumor { background: linear-gradient(135deg, #7f1d1d, #450a0a); border-left: 5px solid #ef4444; padding: 20px; border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

# ── 3. Load Model (Google Drive) ───────────────────────────────
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

# ── 4. Sidebar & Functions ─────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>🧠 NeuroScan AI</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    # FUNCTION 1: Patient Demographics System
    st.markdown("### 📋 1. Patient Details")
    patient_name = st.text_input("Patient Name", "John Doe")
    patient_id = st.text_input("Patient ID", "MRN-84729")
    patient_age = st.number_input("Age", min_value=1, max_value=120, value=45)
    
    st.markdown("---")
    st.markdown("**Developer:** Mohammad Hasnain")

# ── 5. Main Dashboard UI ───────────────────────────────────────
st.markdown('<p class="main-header">Automated Brain Tumor Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced CNN Diagnostics with Patient Tracking</p>', unsafe_allow_html=True)

# Create Tabs for Dashboard vs History
tab1, tab2 = st.tabs(["🔬 Diagnostic Scanner", "🕰️ Session History"])

with tab1:
    with st.spinner("Initializing neural network..."):
        model, model_loaded = load_vision_model()

    if not model_loaded:
        st.error("⚠️ Critical System Error: Unable to load neural network weights.")
        st.stop()

    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.markdown("### 📤 2. Upload MRI Scan")
        uploaded_file = st.file_uploader("Select MRI Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            original_image = Image.open(uploaded_file).convert('RGB')
            
            # FUNCTION 2: Interactive MRI Enhancer
            st.markdown("**🎛️ Image Enhancement Tools**")
            brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
            contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
            
            # Apply Enhancements
            enhancer_b = ImageEnhance.Brightness(original_image)
            img_b = enhancer_b.enhance(brightness)
            enhancer_c = ImageEnhance.Contrast(img_b)
            final_image = enhancer_c.enhance(contrast)
            
            st.image(final_image, caption='Enhanced MRI Preview', use_container_width=True)

    with col2:
        st.markdown("### 🔬 3. Diagnostic Report")
        
        if uploaded_file is None:
            st.info("Awaiting patient data. Please upload an MRI scan on the left.")
        else:
            if st.button("▶ INITIATE AI ANALYSIS", type="primary", use_container_width=True):
                
                progress_bar = st.progress(0)
                time.sleep(0.5)
                progress_bar.progress(50)
                time.sleep(0.5)
                progress_bar.progress(100)
                progress_bar.empty()
                
                # Predict
                img_resized = original_image.resize((150, 150))
                img_array = np.array(img_resized)
                img_array = np.expand_dims(img_array, axis=0) 
                
                prob = model.predict(img_array, verbose=0)[0][0]
                has_tumor = prob >= 0.5
                confidence = prob if has_tumor else (1 - prob)
                conf_percent = confidence * 100
                
                # FUNCTION 3: Clinical Risk Stratification
                risk_level = "LOW RISK"
                if has_tumor and conf_percent > 85:
                    risk_level = "HIGH RISK"
                elif has_tumor and conf_percent <= 85:
                    risk_level = "MODERATE RISK"
                
                # Display Results
                if has_tumor:
                    st.markdown(f"""
                    <div class="result-card-tumor">
                        <h2 style="color: #fca5a5; margin-top: 0;">⚠️ ABNORMALITY DETECTED</h2>
                        <h1 style="color: white; margin-bottom: 0;">{conf_percent:.2f}%</h1>
                        <p style="color: #fca5a5; margin-top: 0; font-size: 0.9rem;">AI CONFIDENCE | <strong>{risk_level}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.warning(f"Patient {patient_name} requires immediate review by a neurologist.")
                    result_text = "TUMOR DETECTED"
                else:
                    st.markdown(f"""
                    <div class="result-card-healthy">
                        <h2 style="color: #6ee7b7; margin-top: 0;">✅ SCAN APPEARS NORMAL</h2>
                        <h1 style="color: white; margin-bottom: 0;">{conf_percent:.2f}%</h1>
                        <p style="color: #6ee7b7; margin-top: 0; font-size: 0.9rem;">AI CONFIDENCE | <strong>{risk_level}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.success(f"No distinct tumor structures detected for {patient_name}.")
                    result_text = "HEALTHY (NO TUMOR)"

                # Save to History (Function 5)
                scan_record = {
                    "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Patient": patient_name,
                    "ID": patient_id,
                    "Diagnosis": result_text,
                    "Confidence": f"{conf_percent:.1f}%"
                }
                st.session_state.scan_history.append(scan_record)

                # FUNCTION 4: Downloadable Medical Report
                report_content = f"""
                ===================================
                NEUROSCAN AI - OFFICIAL REPORT
                ===================================
                Date: {scan_record['Date']}
                Attending AI Developer: Mohammad Hasnain
                
                PATIENT DETAILS
                ---------------
                Name: {patient_name}
                Patient ID: {patient_id}
                Age: {patient_age}
                
                DIAGNOSTIC RESULTS
                ------------------
                AI Diagnosis: {result_text}
                Confidence Score: {conf_percent:.2f}%
                Risk Stratification: {risk_level}
                
                Disclaimer: Investigational use only.
                ===================================
                """
                st.download_button(
                    label="📄 DOWNLOAD PDF/TXT REPORT",
                    data=report_content,
                    file_name=f"{patient_id}_MRI_Report.txt",
                    mime="text/plain",
                    use_container_width=True
                )

with tab2:
    # FUNCTION 5 Display: Session History Tracker
    st.markdown("### 🕰️ Patient Scan History")
    if len(st.session_state.scan_history) == 0:
        st.info("No scans performed in this session yet.")
    else:
        st.table(st.session_state.scan_history)
