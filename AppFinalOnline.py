import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import subprocess
from pathlib import Path
import logging

# ----------------------------------------------------
# LOGGING
# ----------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------
# GITHUB REPO SETTINGS
# ----------------------------------------------------
GITHUB_URL = "https://github.com/Shah-Abdul-Mazid/CapstoneProjectWeb.git"
LOCAL_REPO = Path("CapstoneProjectWeb")

# ----------------------------------------------------
# CLONE OR PULL REPO
# ----------------------------------------------------
def clone_or_pull_repo():
    if LOCAL_REPO.exists():
        st.info("Updating model files from GitHub...")
        try:
            subprocess.run(["git", "-C", str(LOCAL_REPO), "pull"], check=True)
            st.success("Repo updated successfully.")
        except Exception as e:
            st.error(f"Git pull failed: {e}")
            logger.error(e)
    else:
        st.info("Cloning model files from GitHub...")
        try:
            subprocess.run(["git", "clone", GITHUB_URL], check=True)
            st.success("Repo cloned successfully.")
        except Exception as e:
            st.error(f"Git clone failed: {e}")
            logger.error(e)

clone_or_pull_repo()

# ----------------------------------------------------
# MODEL PATH (UPDATED)
# ----------------------------------------------------
MODEL_PATH = LOCAL_REPO / "models" / "DatasetCombined" / "Balance" / "Hybrid_MobDenseNet_CBAM_GradCAM.h5"

st.write(f"Model Path Loaded: `{MODEL_PATH}`")

# ----------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------
@st.cache_resource
def load_mri_model():
    try:
        model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        logger.error(e)
        return None

model = load_mri_model()

# ----------------------------------------------------
# CLASS NAMES
# ----------------------------------------------------
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# ----------------------------------------------------
# GRAD-CAM++ IMPLEMENTATION
# ----------------------------------------------------
def gradcam_plus_plus(model, img_array, layer_name):
    """Grad-CAM++ heatmap generator"""
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        pred_idx = tf.argmax(predictions[0])
        pred_output = predictions[:, pred_idx]

    grads = tape.gradient(pred_output, conv_output)
    grads = tf.maximum(grads, 0)

    weights = tf.reduce_mean(grads, axis=(0, 1, 2))

    cam = tf.reduce_sum(weights * conv_output[0], axis=-1)
    cam = np.maximum(cam, 0)
    cam = cam / (np.max(cam) + 1e-8)

    heatmap = np.uint8(cam * 255)
    heatmap = np.stack([heatmap] * 3, axis=-1)

    return heatmap

# ----------------------------------------------------
# UI
# ----------------------------------------------------
st.title("🧠 Brain MRI Tumor Classification (TensorFlow)")
st.subheader("Model automatically downloaded from GitHub")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded MRI", use_column_width=True)

    # Preprocess
    resized_img = img.resize((224, 224))
    arr = np.expand_dims(np.array(resized_img) / 255.0, axis=0)

    # Predict
    preds = model.predict(arr)[0]
    pred_label = CLASS_NAMES[np.argmax(preds)]

    st.success(f"### 🎯 Prediction: **{pred_label}**")
    st.write(f"Confidence: `{np.max(preds) * 100:.2f}%`")

    # Grad-CAM++
    st.subheader("🔥 Grad-CAM++ Visualization")

    try:
        heatmap = gradcam_plus_plus(model, arr, layer_name="additional_gradcam_layer")

        heatmap = Image.fromarray(heatmap).resize(img.size).convert("RGBA")
        overlay = Image.blend(img.convert("RGBA"), heatmap, alpha=0.45)

        st.image(overlay, caption="Grad-CAM++ Heatmap", use_column_width=True)

    except Exception as e:
        st.error("Grad-CAM++ failed. Incorrect last conv layer name.")
        st.code(str(e))
