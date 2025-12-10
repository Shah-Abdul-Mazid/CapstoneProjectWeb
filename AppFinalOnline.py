# """
# app.py
# Web-based version of your thesis Brain Tumor MRI Classification App.
# Converted from Windows local paths → portable, deployable structure.

# Features:
# ✔ Hybrid model inference (MobileNetV2 + DenseNet121 + CBAM)
# ✔ Grad-CAM++ visualization
# ✔ Academic UI with CSS
# ✔ Prediction box (healthy vs tumor styling)
# ✔ Works on Streamlit Cloud / HuggingFace / Render
# """

# import streamlit as st
# from streamlit_option_menu import option_menu
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from PIL import Image, ImageOps
# import numpy as np
# import cv2
# import io
# import os
# import time
# from typing import Tuple

# # -----------------------------
# # WEB-BASED RELATIVE PATHS
# # -----------------------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# MODEL_PATHS = {
#     "Combined Dataset (Balanced)": os.path.join(BASE_DIR, "models", "Hybrid_MobDenseNet_CBAM_GradCAM.h5")
# }

# # Model classes
# CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# DEFAULT_LAST_CONV_LAYER = "additional_gradcam_layer"

# # ----------------------------------------------------------------------------
# # Streamlit Page Config
# # ----------------------------------------------------------------------------
# st.set_page_config(
#     page_title="Brain Tumor MRI Classification — Web",
#     page_icon="🧠",
#     layout="wide",
# )

# # ----------------------------------------------------------------------------
# # Academic CSS (same as your original)
# # ----------------------------------------------------------------------------
# st.markdown("""
# <style>
# .big-font {font-size:44px !important; font-weight:700; color:#0B4F6C;}
# .title-font {font-size:22px !important; color:#0B3D91;}
# .small {font-size:14px; color:#222; line-height:1.5;}

# .card {
#     background: #FFFFFF;
#     border-radius: 10px;
#     padding: 18px;
#     box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
#     margin-bottom: 12px;
# }

# .footer {
#     text-align:center;
#     padding:14px;
#     background-color:#0B1B2B;
#     color:white;
#     border-radius:8px;
# }
# </style>
# """, unsafe_allow_html=True)

# # ----------------------------------------------------------------------------
# # Model Loading (web optimized)
# # ----------------------------------------------------------------------------
# @st.cache_resource(show_spinner="Loading model…")
# def load_brain_model(model_path: str):
#     return load_model(model_path, compile=False)

# @st.cache_data
# def preprocess_image(img: Image.Image, target_size=(224, 224)):
#     img = ImageOps.fit(img, target_size, Image.LANCZOS)
#     arr = np.asarray(img).astype(np.float32) / 255.0
#     return np.expand_dims(arr, axis=0)

# def get_gradcam_heatmap(model, img_array, last_conv_layer_name):
#     grad_model = tf.keras.models.Model(
#         [model.inputs],
#         [model.get_layer(last_conv_layer_name).output, model.output]
#     )

#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)
#         pred_index = tf.argmax(predictions[0])
#         class_output = predictions[:, pred_index]

#     grads = tape.gradient(class_output, conv_outputs)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

#     conv_outputs = conv_outputs[0].numpy()
#     pooled_grads = pooled_grads.numpy()

#     for i in range(pooled_grads.shape[-1]):
#         conv_outputs[:, :, i] *= pooled_grads[i]

#     heatmap = np.sum(conv_outputs, axis=-1)
#     heatmap = np.maximum(heatmap, 0)
#     heatmap /= np.max(heatmap) + 1e-10

#     return heatmap, int(pred_index.numpy())

# def overlay_heatmap(img_pil, heatmap, alpha=0.5):
#     heatmap_resized = cv2.resize(heatmap, (img_pil.width, img_pil.height))
#     heatmap_uint8 = np.uint8(255 * heatmap_resized)
#     heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

#     img_np = np.array(img_pil)
#     superimposed = cv2.addWeighted(img_np, 1 - alpha, heatmap_color, alpha, 0)
#     return Image.fromarray(superimposed)

# def pil_image_to_bytes(img: Image.Image):
#     buf = io.BytesIO()
#     img.save(buf, format="PNG")
#     return buf.getvalue()

# # ----------------------------------------------------------------------------
# # Sidebar Navigation
# # ----------------------------------------------------------------------------
# with st.sidebar:
#     st.image("assets/brain_icon.png", width=80)
#     selected = option_menu(
#         "Navigation",
#         ["Home", "Model Inference", "Grad-CAM Analysis", "Dataset Preview"],
#         icons=["house", "cpu", "image", "folder"],
#         default_index=0
#     )

# # ----------------------------------------------------------------------------
# # HOME
# # ----------------------------------------------------------------------------
# if selected == "Home":
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown('<div class="big-font">Brain Tumor MRI Classification</div>', unsafe_allow_html=True)
#     st.markdown('<div class="title-font">Hybrid MobileNetV2–DenseNet121 + CBAM + Grad-CAM++</div>', unsafe_allow_html=True)

#     st.write("""
#     This is a web-based deployment of the thesis system for multi-class
#     brain tumor classification using deep learning and attention mechanisms.
#     """)
#     st.image("assets/home_banner.jpg")
#     st.markdown("</div>", unsafe_allow_html=True)

# # ----------------------------------------------------------------------------
# # MODEL INFERENCE
# # ----------------------------------------------------------------------------
# elif selected == "Model Inference":
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown('<div class="title-font">Model Inference</div>', unsafe_allow_html=True)

#     model_choice = st.selectbox("Choose Model", list(MODEL_PATHS.keys()))
#     model = load_brain_model(MODEL_PATHS[model_choice])

#     uploaded = st.file_uploader("Upload MRI Image", type=["png", "jpg", "jpeg"])

#     if uploaded:
#         img = Image.open(uploaded).convert("RGB")
#         st.image(img, caption="Uploaded Image")

#         img_array = preprocess_image(img)

#         start = time.time()
#         preds = model.predict(img_array)[0]
#         infer_time = time.time() - start

#         pred_idx = int(np.argmax(preds))
#         pred_class = CLASS_NAMES[pred_idx]
#         confidence = preds[pred_idx] * 100

#         border = "#0B63D6" if pred_class == "No Tumor" else "#C62828"

#         st.markdown(f"""
#         <div style="padding:18px; border-left:6px solid {border}; border-radius:8px;">
#         <h3>Prediction: {pred_class}</h3>
#         Confidence: {confidence:.2f}%<br>
#         Inference Time: {infer_time:.4f} sec
#         </div>
#         """, unsafe_allow_html=True)

#         st.bar_chart({c: float(p) for c, p in zip(CLASS_NAMES, preds)})

#     st.markdown("</div>", unsafe_allow_html=True)

# # ----------------------------------------------------------------------------
# # GRAD-CAM
# # ----------------------------------------------------------------------------
# elif selected == "Grad-CAM Analysis":
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown('<div class="title-font">Grad-CAM++ Visualization</div>', unsafe_allow_html=True)

#     model_choice = st.selectbox("Select Model", list(MODEL_PATHS.keys()))
#     model = load_brain_model(MODEL_PATHS[model_choice])

#     last_layer = st.text_input("Last Conv Layer", value=DEFAULT_LAST_CONV_LAYER)

#     uploaded = st.file_uploader("Upload MRI", type=["png", "jpg", "jpeg"])

#     if uploaded:
#         img = Image.open(uploaded).convert("RGB")
#         st.image(img, caption="Input MRI")

#         img_array = preprocess_image(img)

#         heatmap, idx = get_gradcam_heatmap(model, img_array, last_layer)
#         gradcam_img = overlay_heatmap(img, heatmap)

#         col1, col2 = st.columns(2)
#         col1.image(img, caption="Original")
#         col2.image(gradcam_img, caption="Grad-CAM++")

#         st.download_button("Download Grad-CAM Image", pil_image_to_bytes(gradcam_img),
#                            file_name="gradcam.png", mime="image/png")

#     st.markdown("</div>", unsafe_allow_html=True)

# # ----------------------------------------------------------------------------
# # DATASET PREVIEW
# # ----------------------------------------------------------------------------
# elif selected == "Dataset Preview":
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown('<div class="title-font">Dataset Samples</div>', unsafe_allow_html=True)

#     SAMPLE_DIR = os.path.join(BASE_DIR, "sample_data")

#     class_dirs = {
#         "Glioma": os.path.join(SAMPLE_DIR, "glioma"),
#         "Meningioma": os.path.join(SAMPLE_DIR, "meningioma"),
#         "No Tumor": os.path.join(SAMPLE_DIR, "no_tumor"),
#         "Pituitary": os.path.join(SAMPLE_DIR, "pituitary"),
#     }

#     for label, folder in class_dirs.items():
#         st.write(f"### {label} Samples")

#         if not os.path.exists(folder):
#             st.warning(f"Missing folder: {folder}")
#             continue

#         cols = st.columns(4)
#         for col, img_name in zip(cols, os.listdir(folder)[:4]):
#             col.image(os.path.join(folder, img_name))

#     st.markdown("</div>", unsafe_allow_html=True)

# # ----------------------------------------------------------------------------
# # FOOTER
# # ----------------------------------------------------------------------------
# st.markdown("""
# <div class="footer">
#     <h4>BSc CSE Thesis Project — 2025</h4>
#     <b>Shah Abdul Mazid</b><br>
#     Hybrid Deep Learning for Brain Tumor Detection with Grad-CAM++
# </div>
# """, unsafe_allow_html=True)

# app.py
# import streamlit as st
# from streamlit_option_menu import option_menu
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from PIL import Image, ImageOps
# import numpy as np
# import cv2
# import io
# import os
# import time
# import gdown
# from typing import Tuple

# # -----------------------------
# # CONFIG
# # -----------------------------
# DEFAULT_LAST_CONV_LAYER = "additional_gradcam_layer"
# CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# # Google Drive File ID for your model
# GDRIVE_FILE_ID = "14DTJgVOMh-34350oyG0EPpJN6qwhMr8e"  # <-- replace with your ID
# MODEL_DIR = "models"
# MODEL_FILENAME = "Hybrid_MobDenseNet_CBAM_GradCAM.h5"
# MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
# MODEL_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

# os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# # PAGE CONFIG
# # -----------------------------
# st.set_page_config(
#     page_title="Brain Tumor MRI Classification",
#     page_icon="🧠",
#     layout="wide"
# )

# # -----------------------------
# # CSS STYLING
# # -----------------------------
# st.markdown("""
# <style>
# .big-font {font-size:44px !important; font-weight:700; color:#0B4F6C;}
# .title-font {font-size:22px !important; color:#0B3D91;}
# .small {font-size:14px; color:#222; line-height:1.5;}
# .card {background: #FFFFFF; border-radius: 10px; padding: 18px;
#        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06); margin-bottom: 12px;}
# .footer {text-align:center; padding:14px; background-color:#0B1B2B;
#          color:white; border-radius:8px;}
# </style>
# """, unsafe_allow_html=True)

# # -----------------------------
# # MODEL UTILITIES
# # -----------------------------
# @st.cache_resource(show_spinner="Downloading and loading model…")
# def load_brain_model(model_path: str = MODEL_PATH, model_url: str = MODEL_URL) -> tf.keras.Model:
#     if not os.path.exists(model_path):
#         st.info("Downloading model from Google Drive…")
#         gdown.download(model_url, model_path, quiet=False)
#         st.success("Model downloaded successfully!")
#     return load_model(model_path, compile=False)

# @st.cache_data
# def preprocess_image(img: Image.Image, target_size=(224, 224)) -> np.ndarray:
#     img = ImageOps.fit(img, target_size, Image.LANCZOS)
#     arr = np.asarray(img).astype(np.float32) / 255.0
#     return np.expand_dims(arr, axis=0)

# def get_gradcam_heatmap(model: tf.keras.Model, img_array: np.ndarray,
#                         last_conv_layer_name: str = DEFAULT_LAST_CONV_LAYER) -> Tuple[np.ndarray, int]:
#     grad_model = tf.keras.models.Model(
#         [model.inputs],
#         [model.get_layer(last_conv_layer_name).output, model.output]
#     )
#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)
#         pred_index = tf.argmax(predictions[0])
#         class_output = predictions[:, pred_index]

#     grads = tape.gradient(class_output, conv_outputs)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     conv_outputs = conv_outputs[0].numpy()
#     pooled_grads = pooled_grads.numpy()
#     for i in range(pooled_grads.shape[-1]):
#         conv_outputs[:, :, i] *= pooled_grads[i]

#     heatmap = np.sum(conv_outputs, axis=-1)
#     heatmap = np.maximum(heatmap, 0)
#     max_val = np.max(heatmap) if np.max(heatmap) != 0 else 1e-10
#     heatmap /= max_val
#     return heatmap, int(pred_index.numpy())

# def overlay_heatmap(img_pil: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> Image.Image:
#     heatmap_resized = cv2.resize(heatmap, (img_pil.width, img_pil.height))
#     heatmap_uint8 = np.uint8(255 * heatmap_resized)
#     heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
#     img_np = np.array(img_pil).astype(np.uint8)
#     superimposed = cv2.addWeighted(img_np, 1 - alpha, heatmap_color, alpha, 0)
#     return Image.fromarray(superimposed)

# def pil_image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
#     buf = io.BytesIO()
#     img.save(buf, format=fmt)
#     return buf.getvalue()

# # -----------------------------
# # SIDEBAR NAVIGATION
# # -----------------------------
# with st.sidebar:
#     st.image("assets/brain_icon.png", width=80)
#     selected = option_menu(
#         "Navigation",
#         ["Home", "Model Inference", "Grad-CAM Analysis", "Dataset Preview"],
#         icons=["house", "cpu", "image", "folder"],
#         default_index=0
#     )

# # -----------------------------
# # HOME PAGE
# # -----------------------------
# if selected == "Home":
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown('<div class="big-font">Brain Tumor MRI Classification</div>', unsafe_allow_html=True)
#     st.markdown('<div class="title-font">Hybrid MobileNetV2–DenseNet121 + CBAM + Grad-CAM++</div>', unsafe_allow_html=True)
#     st.write("Web-based deployment of multi-class brain tumor classification using deep learning and attention mechanisms.")
#     st.image("assets/home_banner.jpg")
#     st.markdown("</div>", unsafe_allow_html=True)

# # -----------------------------
# # MODEL INFERENCE
# # -----------------------------
# elif selected == "Model Inference":
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown('<div class="title-font">Model Inference</div>', unsafe_allow_html=True)

#     model = load_brain_model()
#     uploaded = st.file_uploader("Upload MRI Image", type=["png", "jpg", "jpeg"])

#     if uploaded:
#         img = Image.open(uploaded).convert("RGB")
#         st.image(img, caption="Uploaded Image")
#         img_array = preprocess_image(img)

#         start = time.time()
#         preds = model.predict(img_array)[0]
#         infer_time = time.time() - start

#         pred_idx = int(np.argmax(preds))
#         pred_class = CLASS_NAMES[pred_idx]
#         confidence = preds[pred_idx] * 100
#         border = "#0B63D6" if pred_class == "No Tumor" else "#C62828"

#         st.markdown(f"""
#         <div style="padding:18px; border-left:6px solid {border}; border-radius:8px;">
#         <h3>Prediction: {pred_class}</h3>
#         Confidence: {confidence:.2f}%<br>
#         Inference Time: {infer_time:.4f} sec
#         </div>
#         """, unsafe_allow_html=True)

#         st.bar_chart({c: float(p) for c, p in zip(CLASS_NAMES, preds)})

#     st.markdown("</div>", unsafe_allow_html=True)

# # -----------------------------
# # GRAD-CAM
# # -----------------------------
# elif selected == "Grad-CAM Analysis":
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown('<div class="title-font">Grad-CAM++ Visualization</div>', unsafe_allow_html=True)

#     model = load_brain_model()
#     last_layer = st.text_input("Last Conv Layer", value=DEFAULT_LAST_CONV_LAYER)
#     uploaded = st.file_uploader("Upload MRI", type=["png", "jpg", "jpeg"])

#     if uploaded:
#         img = Image.open(uploaded).convert("RGB")
#         st.image(img, caption="Input MRI")
#         img_array = preprocess_image(img)

#         heatmap, idx = get_gradcam_heatmap(model, img_array, last_layer)
#         gradcam_img = overlay_heatmap(img, heatmap)

#         col1, col2 = st.columns(2)
#         col1.image(img, caption="Original")
#         col2.image(gradcam_img, caption="Grad-CAM++")

#         st.download_button("Download Grad-CAM Image", pil_image_to_bytes(gradcam_img),
#                            file_name="gradcam.png", mime="image/png")

#     st.markdown("</div>", unsafe_allow_html=True)

# # -----------------------------
# # DATASET PREVIEW
# # -----------------------------
# elif selected == "Dataset Preview":
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown('<div class="title-font">Dataset Samples</div>', unsafe_allow_html=True)

#     SAMPLE_DIR = "sample_data"
#     class_dirs = {cls: os.path.join(SAMPLE_DIR, cls.lower()) for cls in CLASS_NAMES}

#     for label, folder in class_dirs.items():
#         st.write(f"### {label} Samples")
#         if not os.path.exists(folder):
#             st.warning(f"Missing folder: {folder}")
#             continue
#         cols = st.columns(4)
#         for col, img_name in zip(cols, os.listdir(folder)[:4]):
#             col.image(os.path.join(folder, img_name))

#     st.markdown("</div>", unsafe_allow_html=True)

# # -----------------------------
# # FOOTER
# # -----------------------------
# st.markdown("""
# <div class="footer">
#     <h4>BSc CSE Thesis Project — 2025</h4>
#     <b>Shah Abdul Mazid</b><br>
#     Hybrid Deep Learning for Brain Tumor Detection with Grad-CAM++
# </div>
# """, unsafe_allow_html=True)


import streamlit as st
from streamlit_option_menu import option_menu

st.title("Brain Tumor MRI Classification App")