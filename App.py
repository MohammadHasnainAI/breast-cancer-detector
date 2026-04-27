"""
Brain Tumor MRI Scanner Web App
Powered by a Custom CNN | Deployable on Streamlit Cloud
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain MRI Scanner",
    page_icon="🧠",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: #0F1117; }
    .hero-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 32px;
        margin-bottom: 24px;
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
    }
    .result-healthy {
        background: linear-gradient(135deg, #0d2b1d, #133d28);
        border: 2px solid #2ecc71;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }
    .result-tumor {
        background: linear-gradient(135deg, #2b0d0d, #3d1313);
        border: 2px solid #e74c3c;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }
    .info-box {
        background: rgba(52,152,219,0.1);
        border: 1px solid #3498db;
        border-radius: 10px;
        padding: 16px;
        margin: 12px 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Model from Google Drive ───────────────────────────────
@st.cache_resource
def load_vision_model():
    try:
        # Check if the model is already downloaded
        if not os.path.exists('brain_tumor_model.h5'):
            # Using your new Brain Tumor Google Drive File ID
            file_id = '13qow6nsxNW0SsiQBPe2QPVnpB3NGB7FN' 
            
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, 'brain_tumor_model.h5', quiet=False)
            
        model = tf.keras.models.load_model('brain_tumor_model.h5')
        return model, True
    except Exception as e:
        return None, False

model, model_loaded = load_vision_model()

# ── UI Header ──────────────────────────────────────────────────
st.markdown("""
<div class="hero-card">
    <h1 style="color:white; font-size:2.4rem; margin:0;">🧠 AI Brain MRI Scanner</h1>
    <p style="color:#aaa; font-size:1.1rem; margin-top:8px;">
        Upload a Brain MRI to detect the presence of a tumor using Deep Learning.
    </p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ Loading Model from Server... Please wait a moment and then refresh the page.")
    st.stop()

# ── Image Uploader ─────────────────────────────────────────────
st.markdown("### 1. Upload MRI Image")
uploaded_file = st.file_uploader("Choose a JPG or PNG file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded MRI Scan', use_container_width=True)
    
    st.markdown("---")
    
    # ── AI Processing ──────────────────────────────────────────
    if st.button("🔍 ANALYZE SCAN", use_container_width=True):
        with st.spinner("AI is analyzing the brain structures..."):
            
            # Format image for our custom CNN (150x150 to match Colab)
            img_resized = image.resize((150, 150))
            img_array = np.array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
            
            # Predict
            prob = model.predict(img_array, verbose=0)[0][0]
            
            # Logic: closer to 1 is Tumor, closer to 0 is Healthy
            has_tumor = prob >= 0.5
            confidence = prob if has_tumor else (1 - prob)
            
            # Output variables
            label = "TUMOR DETECTED" if has_tumor else "HEALTHY (NO TUMOR)"
            css_class = "result-tumor" if has_tumor else "result-healthy"
            emoji = "⚠️" if has_tumor else "✅"
            color = "#e74c3c" if has_tumor else "#2ecc71"
            
            # ── Display Results ────────────────────────────────
            st.markdown(f"""
            <div class="{css_class}">
                <h1 style="color:{color}; font-size:2.5rem; margin:0;">{emoji} {label}</h1>
                <p style="color:white; font-size:1.4rem; margin-top:12px;">
                    AI Confidence Score: <strong>{confidence*100:.1f}%</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if has_tumor:
                st.error("Abnormalities detected. Please consult a neurologist immediately.")
            else:
                st.success("No anomalies detected. Brain structure appears healthy.")

# ── Footer ─────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    ⚠️ <strong>Medical Disclaimer:</strong> This tool is an academic AI project. 
    It is not FDA-approved and cannot replace a professional radiologist.
</div>
""", unsafe_allow_html=True)
