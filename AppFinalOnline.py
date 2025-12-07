import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2
import io
import os
import time
from pathlib import Path
import requests
from tqdm import tqdm

# ==================== MODEL URLS (RAW + LFS WORKS!) ====================
MODEL_PATHS = {
    "Combined Dataset (Balanced)": 
        "https://github.com/Shah-Abdul-Mazid/CapstoneProjectWeb/raw/main/models/DatasetCombined/Balance/Hybrid_MobDenseNet_CBAM_GradCAM.h5",
    
    "Dataset 2 (Balanced)": 
        "https://github.com/Shah-Abdul-Mazid/CapstoneProjectWeb/raw/main/models/Dataset002/Balance/FinalModel/Hybrid_MobDenseNet_CBAM_GradCAM.h5",
    
    "Dataset 1 (Balanced)": 
        "https://github.com/Shah-Abdul-Mazid/CapstoneProjectWeb/raw/main/models/Dataset001/Balance/Hybrid_MobDenseNet_CBAM_GradCAM.h5",
    
    "Combined 3 Datasets (Imbalanced)": 
        "https://github.com/Shah-Abdul-Mazid/CapstoneProjectWeb/raw/main/models/Combine3_dataset/Imbalance/FinalModel/Hybrid_MobDenseNet_CBAM_GradCAM.h5"
}

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
DEFAULT_LAST_CONV_LAYER = "additional_gradcam_layer"

# ==================== SMART MODEL DOWNLOADER (THE FIX!) ====================
@st.cache_resource(show_spinner="Downloading model from GitHub (~300MB, only once)...")
def load_brain_model(url: str):
    cache_dir = Path("cached_models")
    cache_dir.mkdir(exist_ok=True)
    
    filename = url.split("/")[-1]
    local_path = cache_dir / filename
    
    if not local_path.exists():
        st.info(f"First time loading: Downloading {filename}... (1–3 minutes)")
        with st.spinner(f"Downloading {filename}..."):
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total = int(response.headers.get('content-length', 0))
            
            with open(local_path, "wb") as f, tqdm(
                total=total, unit='B', unit_scale=True, unit_divisor=1024,
                desc=filename, leave=False
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        st.success(f"{filename} downloaded and cached!")
    
    return load_model(str(local_path), compile=False)

# ==================== IMAGE PROCESSING ====================
def preprocess_image(img: Image.Image, target_size=(224, 224)):
    img = ImageOps.fit(img, target_size, Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def get_gradcam_heatmap(model, img_array, layer_name=DEFAULT_LAST_CONV_LAYER):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        class_output = predictions[:, class_idx]
    
    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + 1e-8)
    return heatmap.numpy()

def overlay_heatmap(img: Image.Image, heatmap: np.ndarray, alpha=0.6):
    heatmap = cv2.resize(heatmap, (img.width, img.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = heatmap * alpha + np.array(img) * (1 - alpha)
    return Image.fromarray(np.uint8(superimposed))

# ==================== PAGE CONFIG & CSS ====================
st.set_page_config(page_title="Brain Tumor MRI Classification", page_icon="brain", layout="wide")

st.markdown("""
<style>
    .big-font {font-size: 48px !important; font-weight: 800; color: #0B4F6C; text-align: center;}
    .title-font {font-size: 26px !important; color: #1E3A8A; font-weight: 600;}
    .card {background: white; padding: 24px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); margin: 20px 0;}
    .footer {text-align: center; padding: 30px; background: #0B1B2B; color: white; border-radius: 12px; margin-top: 60px;}
    .stButton>button {background:#0B4F6C; color:white; font-weight:bold; border-radius:8px;}
</style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/brain.png", width=120)
    st.markdown("<h2 style='text-align:center; color:#0B4F6C;'>Brain Tumor Classifier</h2>", unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title=None,
        options=["Home", "Methods", "Inference", "Grad-CAM", "Results", "Dataset Preview"],
        icons=["house", "gear", "cpu-fill", "image-fill", "bar-chart-fill", "images"],
        default_index=0,
        styles={"nav-link-selected": {"background-color": "#0B4F6C"}},
    )

# ==================== PAGES ====================
if selected == "Home":
    st.markdown('<p class="big-font">Brain Tumor MRI Classification</p>', unsafe_allow_html=True)
    st.markdown('<p class="title-font" style="text-align:center;">Hybrid MobileNetV2–DenseNet121 with CBAM & Grad-CAM++</p>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 3])
    with col1:
        st.image("https://img.freepik.com/free-photo/doctor-with-mri-scan_23-2149366759.jpg?w=740", caption="MRI Brain Scan")
    with col2:
        st.markdown("""
        <div style="font-size:16px; line-height:1.7;">
        <h3>Key Features</h3>
        <ul>
            <li>Hybrid Architecture: MobileNetV2 + DenseNet121 fusion</li>
            <li>CBAM Attention Mechanism for better tumor focus</li>
            <li>Grad-CAM++ Visual Explainability</li>
            <li>98.7% Accuracy • Multi-class: Glioma, Meningioma, Pituitary, No Tumor</li>
        </ul>
        <p><em>Master's Thesis 2025 • Computer Science & Engineering</em></p>
        </div>
        """, unsafe_allow_html=True)

elif selected in ["Inference", "Grad-CAM"]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"<p class='title-font'>{'Live Inference' if selected=='Inference' else 'Grad-CAM++ Visualization'}</p>", unsafe_allow_html=True)

    model_choice = st.selectbox("Choose Trained Model", options=list(MODEL_PATHS.keys()))
    layer_name = st.text_input("Last Conv Layer Name", value=DEFAULT_LAST_CONV_LAYER)

    uploaded = st.file_uploader("Upload MRI Scan (T1-weighted)", type=["png", "jpg", "jpeg"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img_array = preprocess_image(img)

        # Load model with smart caching
        model = load_brain_model(MODEL_PATHS[model_choice])

        # Prediction
        with st.spinner("Running inference..."):
            start = time.time()
            preds = model.predict(img_array, verbose=0)[0]
            pred_idx = np.argmax(preds)
            pred_class = CLASS_NAMES[pred_idx]
            confidence = preds[pred_idx] * 100
            inference_time = time.time() - start

        # Result
        color = "#16A34A" if pred_class == "No Tumor" else "#DC2626"
        st.markdown(f"""
        <div style="text-align:center; padding:30px; border-radius:16px; background:#F0F9FF; 
                    border-left:12px solid {color}; box-shadow:0 10px 30px rgba(0,0,0,0.15); margin:30px 0;">
            <h1 style="color:{color}; margin:0;">{pred_class}</h1>
            <h3>Confidence: {confidence:.2f}% • Time: {inference_time:.3f}s</h3>
        </div>
        """, unsafe_allow_html=True)

        # Grad-CAM
        try:
            heatmap = get_gradcam_heatmap(model, img_array, layer_name)
            gradcam_img = overlay_heatmap(img, heatmap)

            col1, col2 = st.columns(2)
            col1.image(img, caption="Original MRI")
            col2.image(gradcam_img, caption=f"Grad-CAM++ → {pred_class}")

            buf = io.BytesIO()
            gradcam_img.save(buf, format="PNG")
            st.download_button("Download Grad-CAM", data=buf.getvalue(), file_name=f"gradcam_{pred_class}.png", mime="image/png")
        except Exception as e:
            st.error(f"Grad-CAM failed: {e}")
            st.info("Try layer names: `additional_gradcam_layer`, `top_conv`, `conv5_block16_concat`")

        # Probability chart
        st.bar_chart({name: round(float(p*100), 2) for name, p in zip(CLASS_NAMES, preds)})

    else:
        st.info("Upload an MRI image to start")
        st.image("https://img.freepik.com/free-photo/doctor-holding-mri-brain-scan_23-2149366759.jpg", use_column_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Other pages (Results, Dataset Preview, etc.) — keep your original ones

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <h3>Master's Thesis • Brain Tumor Classification using Hybrid Deep Learning • 2025</h3>
    <p>Built with Streamlit • TensorFlow • Grad-CAM++</p>
</div>
""", unsafe_allow_html=True)