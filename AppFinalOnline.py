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

# ------------------- USE RAW GITHUB URLs (THIS WORKS!) -------------------
MODEL_PATHS = {
    "Combined Dataset (Balanced)": 
        "https://raw.githubusercontent.com/Shah-Abdul-Mazid/CapstoneProjectWeb/main/models/DatasetCombined/Balance/Hybrid_MobDenseNet_CBAM_GradCAM.h5",
    
    "Dataset 2 (Balanced)": 
        "https://raw.githubusercontent.com/Shah-Abdul-Mazid/CapstoneProjectWeb/main/models/Dataset002/Balance/FinalModel/Hybrid_MobDenseNet_CBAM_GradCAM.h5",
    
    "Dataset 1 (Balanced)": 
        "https://raw.githubusercontent.com/Shah-Abdul-Mazid/CapstoneProjectWeb/main/models/Dataset001/Balance/Hybrid_MobDenseNet_CBAM_GradCAM.h5",
    
    "Combined 3 Datasets (Imbalanced)": 
        "https://raw.githubusercontent.com/Shah-Abdul-Mazid/CapstoneProjectWeb/main/models/Combine3_dataset/Imbalance/FinalModel/Hybrid_MobDenseNet_CBAM_GradCAM.h5"
}

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
DEFAULT_LAST_CONV_LAYER = "additional_gradcam_layer"

# ------------------- SMART MODEL DOWNLOADER + CACHE -------------------
@st.cache_resource(show_spinner="Downloading model from GitHub (only once)...")
def load_brain_model_from_github(url: str):
    cache_dir = Path("cached_models")
    cache_dir.mkdir(exist_ok=True)
    
    filename = url.split("/")[-1]
    local_path = cache_dir / filename
    
    # Download only if not already cached
    if not local_path.exists():
        st.info(f"First time: Downloading {filename} (~300–400 MB)... This takes 1–3 minutes.")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total = int(response.headers.get('content-length', 0))
        with open(local_path, "wb") as f, tqdm(
            total=total, unit='B', unit_scale=True, unit_divisor=1024, desc=filename, leave=False
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    # Load the model from local cache
    return load_model(str(local_path), compile=False)

# ------------------- Rest of your functions (unchanged) -------------------
def preprocess_image(img: Image.Image, target_size=(224, 224)) -> np.ndarray:
    img = ImageOps.fit(img, target_size, Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def get_gradcam_heatmap(model, img_array, last_conv_layer_name=DEFAULT_LAST_CONV_LAYER):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_output = predictions[:, pred_index]
    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), int(pred_index.numpy())

def overlay_heatmap(original_img: Image.Image, heatmap: np.ndarray, alpha=0.6):
    heatmap = cv2.resize(heatmap, (original_img.width, original_img.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = heatmap_color * alpha + np.array(original_img) * (1 - alpha)
    return Image.fromarray(np.uint8(superimposed))

# ------------------- Streamlit UI (exactly same as yours) -------------------
st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="brain", layout="wide")

# Your beautiful CSS (keep it)
st.markdown("""
<style>
    .big-font {font-size: 48px !important; font-weight: 800; color: #0B4F6C; text-align: center;}
    .title-font {font-size: 26px !important; color: #1E3A8A; font-weight: 600;}
    .card {background: white; padding: 24px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin: 16px 0;}
    .footer {text-align: center; padding: 20px; background: #0B1B2B; color: white; border-radius: 10px; margin-top: 50px;}
    .stButton>button {background-color: #0B4F6C; color: white; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/brain.png", width=120)
    st.markdown("<h2 style='text-align:center; color:#0B4F6C;'>Brain Tumor Classifier</h2>", unsafe_allow_html=True)
    selected = option_menu(None, ["Home", "Inference", "Grad-CAM", "Results", "Dataset Preview"],
                           icons=["house", "cpu-fill", "image-fill", "bar-chart-fill", "images"], default_index=0)

# Pages (you can keep your original code here — only changed model loading)
if selected == "Inference" or selected == "Grad-CAM":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<p class='title-font'>{'Live Inference' if selected=='Inference' else 'Grad-CAM++ Visualization'}</p>", unsafe_allow_html=True)

    model_choice = st.selectbox("Select Model", options=list(MODEL_PATHS.keys()))
    layer_name = st.text_input("Last Conv Layer Name", value=DEFAULT_LAST_CONV_LAYER)

    uploaded = st.file_uploader("Upload MRI (T1-weighted)", type=["png", "jpg", "jpeg"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img_array = preprocess_image(img)

        model = load_brain_model_from_github(MODEL_PATHS[model_choice])

        preds = model.predict(img_array, verbose=0)[0]
        pred_idx = np.argmax(preds)
        pred_class = CLASS_NAMES[pred_idx]
        confidence = preds[pred_idx] * 100

        # Beautiful result box
        color = "#16A34A" if pred_class == "No Tumor" else "#DC2626"
        st.markdown(f"""
        <div style="text-align:center; padding:30px; border-radius:16px; background:#F0F9FF; 
                    border-left:12px solid {color}; box-shadow:0 10px 30px rgba(0,0,0,0.15); margin:30px 0;">
            <h1 style="color:{color}; margin:0;">{pred_class}</h1>
            <h3>Confidence: {confidence:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)

        # Grad-CAM
        try:
            heatmap, _ = get_gradcam_heatmap(model, img_array, layer_name)
            gradcam_img = overlay_heatmap(img, heatmap)

            col1, col2 = st.columns(2)
            col1.image(img, caption="Original MRI", use_column_width=True)
            col2.image(gradcam_img, caption=f"Grad-CAM++ → {pred_class}", use_column_width=True)

            st.download_button("Download Grad-CAM", data=io.BytesIO(gradcam_img.save(io.BytesIO(), format='PNG')).getvalue(),
                               file_name=f"gradcam_{pred_class}.png", mime="image/png")
        except Exception as e:
            st.error(f"Grad-CAM failed. Try layer name: {e}")

        # Bar chart
        st.bar_chart({name: float(p*100) for name, p in zip(CLASS_NAMES, preds)})

    st.markdown("</div>", unsafe_allow_html=True)

# Keep your other pages (Home, Results, etc.) exactly as they were

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <h3>Master's Thesis • Hybrid MobileNetV2–DenseNet121 + CBAM + Grad-CAM++ • 2025</h3>
</div>
""", unsafe_allow_html=True)