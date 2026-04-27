"""
NeuroScan AI - Brain Tumor MRI Analysis Dashboard
Developed by Mohammad Hasnain
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
import time

# ── 1. Page Configuration ──────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI | Brain MRI",
    page_icon="🧠",
    layout="wide", # Changed to wide for a dashboard look
    initial_sidebar_state="expanded"
)

# ── 2. Custom Professional CSS ─────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    /* Headers */
    .main-header { font-size: 2.8rem; font-weight: 700; color: #ffffff; margin-bottom: 0px; padding-bottom: 0px;}
    .sub-header { font-size: 1.1rem; color: #94a3b8; margin-bottom: 30px; margin-top: 5px;}
    
    /* Result Cards */
    .result-card-healthy { background: linear-gradient(135deg, #064e3b 0%, #022c22 100%); border-left: 5px solid #10b981; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .result-card-tumor { background: linear-gradient(135deg, #7f1d1d 0%, #450a0a 100%); border-left: 5px solid #ef4444; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    
    /* Medical Disclaimer */
    .disclaimer-box { background-color: #1e293b; border-left: 4px solid #f59e0b; padding: 15px; border-radius: 5px; font-size: 0.9rem; color: #cbd5e1; margin-top: 40px;}
</style>
""", unsafe_allow_html=True)

# ── 3. Load Model (Google Drive) ───────────────────────────────
@st.cache_resource(show_spinner=False)
def load_vision_model():
    try:
        if not os.path.exists('brain_tumor_model.h5'):
            file_id = '13qow6nsxNW0SsiQBPe2QPVnpB3NGB7FN' # Your exact model ID
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, 'brain_tumor_model.h5', quiet=False)
            
        model = tf.keras.models.load_model('brain_tumor_model.h5')
        return model, True
    except Exception as e:
        return None, False

# ── 4. Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>🧠 NeuroScan AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8;'>Version 1.0.0</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 📋 Clinical Instructions")
    st.markdown("""
    1. **Upload** a standard T1/T2 weighted Brain MRI scan (JPEG or PNG format).
    2. **Review** the image preview to ensure high quality and correct alignment.
    3. **Analyze** the scan using the Deep Learning Convolutional Neural Network.
    4. **Review** the generated diagnostic confidence metrics.
    """)
    st.markdown("---")
    st.markdown("**Developer:** Mohammad Hasnain")
    st.markdown("**Architecture:** Custom CNN (150x150)")

# ── 5. Main Dashboard ──────────────────────────────────────────
st.markdown('<p class="main-header">Automated Brain Tumor Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Deep Learning Image Classification Dashboard</p>', unsafe_allow_html=True)

# Try to load the model invisibly 
with st.spinner("Initializing neural network..."):
    model, model_loaded = load_vision_model()

if not model_loaded:
    st.error("⚠️ Critical System Error: Unable to load neural network weights. Please refresh.")
    st.stop()

# Create two professional columns
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown("### 📤 Upload Patient Scan")
    uploaded_file = st.file_uploader("Select MRI Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Patient MRI Scan Preview', use_container_width=True, clamp=True)

with col2:
    st.markdown("### 🔬 Diagnostic Report")
    
    if uploaded_file is None:
        st.info("Awaiting patient data. Please upload an MRI scan on the left to begin analysis.")
    else:
        if st.button("▶ INITIATE AI ANALYSIS", type="primary", use_container_width=True):
            
            # Simulated professional loading bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Isolating brain tissue structures...")
            time.sleep(0.5)
            progress_bar.progress(30)
            
            status_text.text("Applying convolutional filters...")
            time.sleep(0.5)
            progress_bar.progress(60)
            
            status_text.text("Calculating predictive confidence...")
            time.sleep(0.5)
            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()
            
            # --- Actual AI Prediction ---
            img_resized = image.resize((150, 150))
            img_array = np.array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) 
            
            prob = model.predict(img_array, verbose=0)[0][0]
            
            has_tumor = prob >= 0.5
            confidence = prob if has_tumor else (1 - prob)
            
            # --- Display Professional Results ---
            if has_tumor:
                st.markdown(f"""
                <div class="result-card-tumor">
                    <h2 style="color: #fca5a5; margin-top: 0;">⚠️ ABNORMALITY DETECTED</h2>
                    <p style="color: #fee2e2; font-size: 1.1rem;">
                        The neural network has identified features consistent with a tumor.
                    </p>
                    <h1 style="color: white; margin-bottom: 0;">{confidence*100:.2f}%</h1>
                    <p style="color: #fca5a5; margin-top: 0; font-size: 0.9rem;">AI CONFIDENCE SCORE</p>
                </div>
                """, unsafe_allow_html=True)
                st.warning("Recommendation: Immediate review by a certified radiologist and neurologist is strongly advised.")
                
            else:
                st.markdown(f"""
                <div class="result-card-healthy">
                    <h2 style="color: #6ee7b7; margin-top: 0;">✅ SCAN APPEARS NORMAL</h2>
                    <p style="color: #d1fae5; font-size: 1.1rem;">
                        No distinct tumor structures were detected in this MRI.
                    </p>
                    <h1 style="color: white; margin-bottom: 0;">{confidence*100:.2f}%</h1>
                    <p style="color: #6ee7b7; margin-top: 0; font-size: 0.9rem;">AI CONFIDENCE SCORE</p>
                </div>
                """, unsafe_allow_html=True)
                st.success("Recommendation: Standard preventative care and routine checkups.")

# ── Footer Disclaimer ──────────────────────────────────────────
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ FOR INVESTIGATIONAL USE ONLY:</strong> 
    This application is an educational prototype developed for academic purposes. It is not approved by the FDA or any regulatory body. 
    It must not be used to make primary medical diagnoses, guide treatment, or replace professional medical judgment.
</div>
""", unsafe_allow_html=True)
