"""
Breast Cancer X-Ray (Mammogram) Scanner
Powered by Deep Learning CNN | Deployable on Streamlit Cloud
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Mammogram Scanner",
    page_icon="🩻",
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
    .result-benign {
        background: linear-gradient(135deg, #0d2b1d, #133d28);
        border: 2px solid #2ecc71;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }
    .result-malignant {
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

# ── Load Model ─────────────────────────────────────────────────
@st.cache_resource
def load_vision_model():
    try:
        model = tf.keras.models.load_model('mammogram_model.h5')
        return model, True
    except Exception as e:
        return None, False

model, model_loaded = load_vision_model()

# ── UI Header ──────────────────────────────────────────────────
st.markdown("""
<div class="hero-card">
    <h1 style="color:white; font-size:2.4rem; margin:0;">🩻 AI Mammogram Scanner</h1>
    <p style="color:#aaa; font-size:1.1rem; margin-top:8px;">
        Upload a Breast X-Ray to detect signs of Malignancy using Deep Learning.
    </p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ Model not found! Please upload 'mammogram_model.h5' to your GitHub.")
    st.stop()

# ── Image Uploader ─────────────────────────────────────────────
st.markdown("### 1. Upload X-Ray Image")
uploaded_file = st.file_uploader("Choose a JPG or PNG file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Mammogram', use_container_width=True)
    
    st.markdown("---")
    
    # ── AI Processing ──────────────────────────────────────────
    if st.button("🔍 ANALYZE X-RAY", use_container_width=True):
        with st.spinner("AI is analyzing the cellular structures..."):
            
            # Format image for the CNN (224x224)
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
            
            # Predict
            prob = model.predict(img_array, verbose=0)[0][0]
            
            # Logic: closer to 1 is Malignant, closer to 0 is Benign
            is_malignant = prob >= 0.5
            confidence = prob if is_malignant else (1 - prob)
            
            # Output variables
            label = "MALIGNANT DETECTED" if is_malignant else "BENIGN (NO CANCER)"
            css_class = "result-malignant" if is_malignant else "result-benign"
            emoji = "🔴" if is_malignant else "🟢"
            color = "#e74c3c" if is_malignant else "#2ecc71"
            
            # ── Display Results ────────────────────────────────
            st.markdown(f"""
            <div class="{css_class}">
                <h1 style="color:{color}; font-size:2.5rem; margin:0;">{emoji} {label}</h1>
                <p style="color:white; font-size:1.4rem; margin-top:12px;">
                    AI Confidence Score: <strong>{confidence*100:.1f}%</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if is_malignant:
                st.error("⚠️ Suspicious features detected. Immediate biopsy or medical consultation is recommended.")
            else:
                st.success("✅ Image appears benign. Regular screening is still recommended.")

# ── Footer ─────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    ⚠️ <strong>Medical Disclaimer:</strong> This tool is an academic AI project. 
    It is not FDA-approved and cannot replace a professional radiologist.
</div>
""", unsafe_allow_html=True)
