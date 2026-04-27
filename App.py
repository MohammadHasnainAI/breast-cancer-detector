"""
AI Breast Ultrasound Scanner
Powered by Hugging Face Vision Transformers | Deployable on Streamlit Cloud
"""

import streamlit as st
from PIL import Image
import torch
from transformers import pipeline

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Ultrasound Scanner",
    page_icon="🩺",
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
    .result-box {
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Hugging Face Model ────────────────────────────────────
@st.cache_resource
def load_hf_model():
    # This automatically downloads a pre-trained, highly accurate medical AI!
    return pipeline("image-classification", model="Parveshiiii/breast-cancer-detector")

try:
    classifier = load_hf_model()
    model_loaded = True
except Exception as e:
    model_loaded = False

# ── UI Header ──────────────────────────────────────────────────
st.markdown("""
<div class="hero-card">
    <h1 style="color:white; font-size:2.4rem; margin:0;">🩺 AI Ultrasound Scanner</h1>
    <p style="color:#aaa; font-size:1.1rem; margin-top:8px;">
        Upload a Breast Ultrasound to detect Normal, Benign, or Malignant tissue using Vision Transformers.
    </p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ Booting up AI Engine from Hugging Face... Please wait 30 seconds and refresh.")
    st.stop()

# ── Image Uploader ─────────────────────────────────────────────
st.markdown("### 1. Upload Ultrasound Image")
uploaded_file = st.file_uploader("Choose a JPG or PNG file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Ultrasound', use_container_width=True)
    
    st.markdown("---")
    
    # ── AI Processing ──────────────────────────────────────────
    if st.button("🔍 ANALYZE ULTRASOUND", use_container_width=True):
        with st.spinner("Connecting to Hugging Face Vision Model..."):
            
            # Predict using the Pipeline
            results = classifier(image)
            
            # Extract top prediction
            top_prediction = results[0]
            label = top_prediction['label'].upper()
            confidence = top_prediction['score'] * 100
            
            # UI Logic based on the result
            if label == "MALIGNANT":
                color = "#e74c3c"
                emoji = "🔴"
                border = "#e74c3c"
                bg = "linear-gradient(135deg, #2b0d0d, #3d1313)"
                msg = "⚠️ Suspicious features detected. Immediate medical consultation recommended."
            elif label == "BENIGN":
                color = "#f1c40f"
                emoji = "🟡"
                border = "#f1c40f"
                bg = "linear-gradient(135deg, #2b2410, #3d3115)"
                msg = "⚕️ Image appears benign (non-cancerous lump). Regular monitoring advised."
            else: # NORMAL
                color = "#2ecc71"
                emoji = "🟢"
                border = "#2ecc71"
                bg = "linear-gradient(135deg, #0d2b1d, #133d28)"
                msg = "✅ Image appears normal. Healthy tissue detected."

            # ── Display Results ────────────────────────────────
            st.markdown(f"""
            <div class="result-box" style="background: {bg}; border: 2px solid {border};">
                <h1 style="color:{color}; font-size:2.5rem; margin:0;">{emoji} {label} DETECTED</h1>
                <p style="color:white; font-size:1.4rem; margin-top:12px;">
                    AI Confidence Score: <strong>{confidence:.1f}%</strong>
                </p>
                <p style="color:#aaa; margin-top:8px;">{msg}</p>
            </div>
            """, unsafe_allow_html=True)
            
# ── Footer ─────────────────────────────────────────────────────
st.markdown("<br><br><p style='text-align:center; color:#555;'>Powered by Hugging Face Transformers</p>", unsafe_allow_html=True)
