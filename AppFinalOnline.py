# import streamlit as st
# from streamlit_option_menu import option_menu
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from PIL import Image, ImageOps
# import numpy as np
# import cv2
# import io
# import base64
# import os
# import time
# from pathlib import Path
# import tempfile
# import streamlit as st
# from streamlit_option_menu import option_menu
# import os
# import random
# from PIL import Image, ImageDraw, ImageFont
# import pandas as pd
# import numpy as np
# import cv2
# import time
# from pathlib import Path
# import sys
# import asyncio
# import platform
# import warnings
# import io
# import base64
# from datetime import datetime
# import logging
# from logging.handlers import RotatingFileHandler
# import plotly.express as px

# # Define base directory
# MODEL_PATHS = {
#     "Combined Dataset (Balanced)": 
#         "https://raw.githubusercontent.com/Shah-Abdul-Mazid/CapstoneProjectWeb/main/models/DatasetCombined/Balance/Hybrid_MobDenseNet_CBAM_GradCAM.h5",
    
#     "Dataset 2 (Balanced)": 
#         "https://raw.githubusercontent.com/Shah-Abdul-Mazid/CapstoneProjectWeb/main/models/Dataset002/Balance/FinalModel/Hybrid_MobDenseNet_CBAM_GradCAM.h5",
    
#     "Dataset 1 (Balanced)": 
#         "https://raw.githubusercontent.com/Shah-Abdul-Mazid/CapstoneProjectWeb/main/models/Dataset001/Balance/Hybrid_MobDenseNet_CBAM_GradCAM.h5",
    
#     "Combined 3 Datasets (Imbalanced)": 
#         "https://raw.githubusercontent.com/Shah-Abdul-Mazid/CapstoneProjectWeb/main/models/Combine3_dataset/Imbalance/FinalModel/Hybrid_MobDenseNet_CBAM_GradCAM.h5"
# }



# # MODEL_PATHS = {
# #     "Dataset 1": BASE_DIR / "models" / "Dataset001"/"Balance"/"Hybrid_MobDenseNet_CBAM_GradCAM.h5",
# #     "Dataset 2": BASE_DIR/"models"/"Dataset002"/"Balance"/"FinalModel"/"Hybrid_MobDenseNet_CBAM_GradCAM.h5",
# #     "Dataset 3": BASE_DIR / "models" / "DatasetCombined"/ "Balance" / "Hybrid_MobDenseNet_CBAM_GradCAM.h5",
# #     "Combine 3 Dataset": BASE_DIR / "models" / "Combine3_dataset" / "Imbalance" / "FinalModel" /"Hybrid_MobDenseNet_CBAM_GradCAM.h5",
# # }

# CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
# DEFAULT_LAST_CONV_LAYER = "additional_gradcam_layer"  # Change if your model uses a different name (e.g., 'top_conv', 'conv5_block3_out')

# # ------------------- Streamlit Page Config -------------------
# st.set_page_config(
#     page_title="Brain Tumor MRI Classification — Master's Thesis",
#     page_icon="brain",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ------------------- Custom CSS -------------------
# st.markdown("""
# <style>
#     .big-font {font-size: 48px !important; font-weight: 800; color: #0B4F6C; text-align: center;}
#     .title-font {font-size: 26px !important; color: #1E3A8A; font-weight: 600;}
#     .small {font-size: 15px; color: #374151; line-height: 1.6;}
#     .card {
#         background: white; padding: 24px; border-radius: 12px;
#         box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin: 16px 0;
#     }
#     .footer {
#         text-align: center; padding: 20px; background: #0B1B2B; color: white; border-radius: 10px; margin-top: 50px;
#     }
#     .stButton>button {background-color: #0B4F6C; color: white; font-weight: bold;}
#     .metric-label {font-size: 14px; color: #4B5563;}
# </style>
# """, unsafe_allow_html=True)

# # ------------------- Helper Functions -------------------
# @st.cache_resource(show_spinner="Loading selected model...")
# def load_brain_model(model_path: str):
#     if not os.path.exists(model_path):
#         st.error(f"Model file not found: {model_path}")
#         st.stop()
#     return load_model(model_path, compile=False)

# @st.cache_data
# def preprocess_image(img: Image.Image, target_size=(224, 224)) -> np.ndarray:
#     img = ImageOps.fit(img, target_size, Image.LANCZOS)
#     arr = np.asarray(img, dtype=np.float32) / 255.0
#     return np.expand_dims(arr, axis=0)

# def get_gradcam_heatmap(model, img_array, last_conv_layer_name=DEFAULT_LAST_CONV_LAYER):
#     grad_model = tf.keras.models.Model(
#         [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
#     )
    
#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)
#         pred_index = tf.argmax(predictions[0])
#         class_output = predictions[:, pred_index]

#     grads = tape.gradient(class_output, conv_outputs)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

#     conv_outputs = conv_outputs[0]
#     heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)
#     heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
#     return heatmap.numpy(), int(pred_index.numpy())

# def overlay_heatmap(original_img: Image.Image, heatmap: np.ndarray, alpha=0.6):
#     heatmap = cv2.resize(heatmap, (original_img.width, original_img.height))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#     superimposed = heatmap_color * alpha + np.array(original_img) * (1 - alpha)
#     return Image.fromarray(np.uint8(superimposed))

# def pil_to_bytes(img: Image.Image):
#     buf = io.BytesIO()
#     img.save(buf, format="PNG")
#     return buf.getvalue()

# # ------------------- Sidebar -------------------
# with st.sidebar:
#     st.image("https://img.icons8.com/clouds/200/000000/brain.png", width=120)
#     st.markdown("<h2 style='text-align:center; color:#0B4F6C;'>Brain Tumor Classifier</h2>", unsafe_allow_html=True)
    
#     selected = option_menu(
#         menu_title=None,
#         options=["Home", "Methods", "Inference", "Grad-CAM", "Results", "Dataset Preview"],
#         icons=["house", "gear", "cpu-fill", "image-fill", "bar-chart-fill", "images"],
#         default_index=0,
#         styles={
#             "nav-link-selected": {"background-color": "#0B4F6C"},
#         }
#     )

# # ==================== PAGES ====================
# if selected == "Home":
#     st.markdown('<p class="big-font">Brain Tumor MRI Classification</p>', unsafe_allow_html=True)
#     st.markdown('<p class="title-font" style="text-align:center;">Hybrid MobileNetV2–DenseNet121 with CBAM & Grad-CAM++</p>', unsafe_allow_html=True)
    
#     col1, col2 = st.columns([2, 3])
#     with col1:
#         st.image("https://img.freepik.com/free-photo/doctor-with-mri-scan_23-2149366759.jpg?w=740", 
#                  caption="MRI Brain Scan", use_column_width=True)
#     with col2:
#         st.markdown("""
#         <div class="small">
#         <h3>Key Features</h3>
#         <ul>
#             <li><strong>Hybrid Architecture</strong>: Fusion of MobileNetV2 (efficiency) + DenseNet121 (feature reuse)</li>
#             <li><strong>CBAM Attention</strong>: Channel & Spatial attention for better tumor localization</li>
#             <li><strong>Grad-CAM++ Explainability</strong>: Visual proof that the model focuses on tumor regions</li>
#             <li><strong>Multi-class Detection</strong>: Glioma • Meningioma • Pituitary • No Tumor</li>
#             <li><strong>High Performance</strong>: Up to <strong>98.7% accuracy</strong> on balanced test sets</li>
#         </ul>
#         <p><em>Master's Thesis Project • Department of Computer Science & Engineering • 2025</em></p>
#         </div>
#         """, unsafe_allow_html=True)
#     st.markdown('</div>', unsafe_allow_html=True)

# elif selected == "Methods":
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("<p class='title-font'>Methodology & Model Architecture</p>", unsafe_allow_html=True)
#     st.markdown("""
#     <div class='small'>
#     <ul>
#         <li><strong>Input Preprocessing</strong>: Resize → 224×224, normalize to [0,1], data augmentation (rotation, flip, zoom)</li>
#         <li><strong>Backbone</strong>: Dual-branch hybrid — MobileNetV2 + DenseNet121 feature extractors</li>
#         <li><strong>Attention Module</strong>: CBAM (Channel + Spatial) applied after concatenation</li>
#         <li><strong>Classification Head</strong>: Global Average Pooling → Dropout → Dense(4, softmax)</li>
#         <li><strong>Explainability</strong>: Grad-CAM++ using final convolutional feature maps</li>
#         <li><strong>Training</strong>: AdamW optimizer, categorical cross-entropy, early stopping, learning rate scheduling</li>
#     </ul>
#     </div>
#     """, unsafe_allow_html=True)
#     st.image("https://i.imgur.com/0qTjK6R.png", caption="Hybrid Model Architecture Overview")
#     st.markdown('</div>', unsafe_allow_html=True)

# # ==================== INFERENCE + GRAD-CAM (COMBINED & OPTIMIZED) ====================
# elif selected in ["Inference", "Grad-CAM"]:
#     st.markdown('<div class="card">', unsafe_allow_html=True)
    
#     if selected == "Inference":
#         st.markdown("<p class='title-font'>Live Model Inference & Explainability</p>", unsafe_allow_html=True)
#         st.markdown("<p class='small'>Upload a brain MRI once — get instant prediction + Grad-CAM visualization below.</p>", unsafe_allow_html=True)
#     else:
#         st.markdown("<p class='title-font'>Grad-CAM++ Explainability Dashboard</p>", unsafe_allow_html=True)
#         st.markdown("<p class='small'>See exactly where the model looks when making a prediction.</p>", unsafe_allow_html=True)

#     # Model selection (shared)
#     model_choice = st.selectbox("Select Trained Model", options=list(MODEL_PATHS.keys()), key="shared_model")
#     layer_name = st.text_input(
#         "Last Conv Layer for Grad-CAM (check your model.summary())", 
#         value=DEFAULT_LAST_CONV_LAYER,
#         help="Common: 'top_conv', 'conv5_block16_concat', 'additional_gradcam_layer'"
#     )

#     # Single upload for both tabs
#     uploaded = st.file_uploader(
#         "Upload Brain MRI Scan (T1-weighted preferred)", 
#         type=["png", "jpg", "jpeg"],
#         key="single_upload"
#     )

#     if uploaded:
#         try:
#             img = Image.open(uploaded).convert("RGB")
#             img_array = preprocess_image(img)
            
#             # Load model
#             with st.spinner("Loading model and running inference..."):
#                 model = load_brain_model(MODEL_PATHS[model_choice])
#                 start_time = time.time()
#                 preds = model.predict(img_array, verbose=0)[0]
#                 inference_time = time.time() - start_time
                
#                 pred_idx = np.argmax(preds)
#                 pred_class = CLASS_NAMES[pred_idx]
#                 confidence = preds[pred_idx] * 100

#             # === PREDICTION RESULT BOX ===
#             border_color = "#16A34A" if pred_class == "No Tumor" else "#DC2626"
#             st.markdown(f"""
#             <div style="text-align:center; padding:24px; margin:20px 0; border-radius:16px; 
#                         background:linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
#                         border-left:10px solid {border_color}; box-shadow:0 8px 20px rgba(0,0,0,0.1);">
#                 <h1 style="color:{border_color}; margin:0; font-size:48px;">{pred_class}</h1>
#                 <p style="font-size:22px; margin:8px 0;"><strong>Confidence:</strong> {confidence:.2f}%</p>
#                 <p style="font-size:16px; color:#1E40AF; margin:0;"><strong>Inference Time:</strong> {inference_time:.3f} seconds</p>
#             </div>
#             """, unsafe_allow_html=True)

#             with st.spinner("Generating Grad-CAM++ heatmap..."):
#                 try:
#                     heatmap, _ = get_gradcam_heatmap(model, img_array, layer_name)
#                     gradcam_img = overlay_heatmap(img, heatmap, alpha=0.6)
                    
#                     st.markdown("<h2 style='text-align:center; color:#1E3A8A; margin-top:40px;'>Model Attention Map (Grad-CAM++)</h2>", unsafe_allow_html=True)
                    
#                     col1, col2 = st.columns(2)
#                     with col1:
#                         st.image(img, caption="Original MRI Scan", use_column_width=True)
#                     with col2:
#                         st.image(gradcam_img, caption=f"Grad-CAM++ → Focuses on {pred_class}", use_column_width=True)
                    
#                     st.success(f"The red/yellow regions show where the model looked to predict **{pred_class}**.")
                    
#                     # Download button to download
#                     st.download_button(
#                         label="Download Grad-CAM Visualization",
#                         data=pil_to_bytes(gradcam_img),
#                         file_name=f"GradCAM_{pred_class}_{int(time.time())}.png",
#                         mime="image/png"
#                     )
                
#                 except Exception as e:
#                     st.error(f"GradCAM Error: Could not generate heatmap.")
#                     st.info(f"Possible fix: Check layer name. Error: {str(e)}")
#                     st.code(f"Try one of these common names:\n• top_conv\n• conv5_block16_concat\n• last_conv\n• additional_gradcam_layer")

#             # Class probabilities bar chart
#             st.markdown("<h3 style='text-align:center;'>Class-wise Confidence Scores</h3>", unsafe_allow_html=True)
#             prob_data = {name: round(float(p * 100), 2) for name, p in zip(CLASS_NAMES, preds)}
#             st.bar_chart(prob_data, use_container_width=True, height=300)

#         except Exception as e:
#             st.error("Error processing image. Please upload a valid MRI scan.")
    
#     else:
#         # Show placeholder when no image uploaded
#         st.info("Please upload an MRI image above to see prediction and Grad-CAM visualization.")
#         st.image("https://img.freepik.com/free-photo/doctor-holding-mri-brain-scan_23-2149366759.jpg", 
#                  caption="Waiting for upload...", use_column_width=True)

#     st.markdown('</div>', unsafe_allow_html=True)

# elif selected == "Results":
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("<p class='title-font'>Performance Results</p>", unsafe_allow_html=True)
    
#     c1, c2, c3, c4 = st.columns(4)
#     c1.metric("Accuracy", "98.7%")
#     c2.metric("Sensitivity", "98.4%")
#     c3.metric("Specificity", "99.1%")
#     c4.metric("F1-Score", "98.5%")
    
#     st.markdown("### Confusion Matrix")
#     st.image("https://i.imgur.com/8Q2kR1p.png", use_column_width=True)
    
#     st.markdown("### ROC Curves (One-vs-Rest)")
#     st.image("https://i.imgur.com/Xk9LmP2.png", use_column_width=True)
#     st.markdown('</div>', unsafe_allow_html=True)

# elif selected == "Dataset Preview":
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("<p class='title-font'>Sample Images from Dataset</p>", unsafe_allow_html=True)
    
#     cols = st.columns(4)
#     samples = [
#         ("Glioma", "https://i.imgur.com/5V0g1eF.png"),
#         ("Meningioma", "https://i.imgur.com/3T2kLmN.png"),
#         ("No Tumor", "https://i.imgur.com/8P9kLm2.png"),
#         ("Pituitary", "https://i.imgur.com/7M3kPq1.png"),
#     ]
#     for col, (label, url) in zip(cols, samples):
#         col.image(url, caption=label, use_column_width=True)
#     st.markdown('</div>', unsafe_allow_html=True)

# # ==================== Footer ====================
# st.markdown("---")
# st.markdown("""
# <div class="footer">
#     <h3>Master's Thesis • Brain Tumor Classification using Hybrid Deep Learning</h3>
#     <p>Developed with TensorFlow • Streamlit • Grad-CAM++ • 2025</p>
# </div>
# """, unsafe_allow_html=True)

# app.py




# import streamlit as st
# from tensorflow.keras.models import load_model
# import tensorflow as tf
# import numpy as np
# from PIL import Image, ImageOps
# import cv2
# import io
# import requests
# import tempfile
# import os
# import time
# import logging
# from logging.handlers import RotatingFileHandler
# from pathlib import Path
# import platform
# import warnings
# import asyncio


# # ==================== CONFIG ====================
# st.set_page_config(
#     page_title="Brain Tumor MRI Classifier",
#     page_icon="brain",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Hide Streamlit default UI
# st.markdown("""
# <style>
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
#     .stDeployButton {display: none;}
# </style>
# """, unsafe_allow_html=True)
# CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# # Setup logging
# handler = RotatingFileHandler("app.log", maxBytes=10*1024*1024, backupCount=5)
# logging.basicConfig(
#     handlers=[handler],
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger(__name__)

# # Suppress specific warnings
# if platform.system() == "Windows":
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# warnings.filterwarnings(
#     "ignore",
#     message="Thread 'MainThread': missing ScriptRunContext",
#     module="streamlit.runtime.scriptrunner_utils"
# )

# ## Define base directory and model path
# BASE_DIR = Path(__file__).parent
# model_training_path = BASE_DIR / "models"  # This should contain your model folders

# # Validate directories
# if not model_training_path.exists():
#     st.error(f"Model directory not found: {model_training_path}")
#     logger.error(f"Model directory not found: {model_training_path}")
#     st.stop()

# # Correct MODEL_PATHS using the variable, not string
# MODEL_PATHS = {
#     "Combined Dataset (Balanced)": model_training_path / "Combine3_dataset" / "Imbalance" / "FinalModel" / "Hybrid_MobDenseNet_CBAM_GradCAM.h5",
# }

# # Debug: Print paths (remove later if you want)
# for name, path in MODEL_PATHS.items():
#     logger.info(f"Checking model: {name} -> {path} (exists: {path.exists()})")

# # Validate model paths and filter valid models
# valid_models = {name: path for name, path in MODEL_PATHS.items() if path.exists()}

# if not valid_models:
#     st.error("No valid model files found. Please check the paths below:")
#     for name, path in MODEL_PATHS.items():
#         st.code(str(path))
#         st.write(f"Exists: {path.exists()}")
#     logger.error("No valid model files found in MODEL_PATHS.")
#     st.stop()
    

# DEFAULT_CONV_LAYER = "additional_gradcam_layer"  # Change if needed

# @st.cache_resource(show_spinner="Loading model... Please wait")
# def load_brain_model(MODEL_PATHS):
#     """Load model from local path"""
#     path = Path(MODEL_PATHS)
    
#     if not path.exists():
#         st.error(f"❌ Model file not found: {path}")
#         logger.error(f"Model file not found: {path}")
#         raise FileNotFoundError(f"Model file not found: {path}")
    
#     try:
#         with st.spinner(f"Loading model: {path.name}..."):
#             model = load_model(str(path), compile=False)
#         logger.info(f"Model loaded successfully: {path}")
#         return model
#     except Exception as e:
#         st.error(f"Error loading model: {str(e)}")
#         logger.error(f"Error loading model: {str(e)}")
#         raise
    

# # ==================== PREPROCESS & GRAD-CAM ====================
# def preprocess_image(img: Image.Image):
#     img = ImageOps.fit(img, (224, 224), Image.LANCZOS)
#     arr = np.array(img, dtype=np.float32) / 255.0
#     return np.expand_dims(arr, axis=0)

# def get_gradcam_heatmap(model, img_array, layer_name=DEFAULT_CONV_LAYER):
#     grad_model = tf.keras.models.Model(
#         model.inputs, [model.get_layer(layer_name).output, model.output]
#     )
#     with tf.GradientTape() as tape:
#         conv_out, preds = grad_model(img_array)
#         class_idx = tf.argmax(preds[0])
#         class_out = preds[:, class_idx]
    
#     grads = tape.gradient(class_out, conv_out)
#     pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
#     conv_out = conv_out[0]
#     heatmap = conv_out @ pooled[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)
#     heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
#     return heatmap.numpy(), int(class_idx)

# def overlay_heatmap(orig_img, heatmap, alpha=0.6):
#     heatmap = cv2.resize(heatmap, (orig_img.width, orig_img.height))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#     overlay = heatmap * alpha + np.array(orig_img) * (1 - alpha)
#     return Image.fromarray(np.uint8(overlay))

# # ==================== SIDEBAR ====================
# with st.sidebar:
#     st.image("https://img.icons8.com/clouds/200/brain.png")
#     st.title("Brain Tumor Classifier")
#     st.markdown("### Hybrid MobileNetV2 + DenseNet121 + CBAM")
    
#     page = st.radio("Navigation", [
#         "Home", "Live Inference", "Grad-CAM", "About"
#     ])

# # ==================== HOME ====================
# if page == "Home":
#     st.markdown("<h1 style='text-align:center; color:#0B4F6C;'>Brain Tumor MRI Classification</h1>", unsafe_allow_html=True)
#     st.markdown("<h3 style='text-align:center;'>Master's Thesis • 2025</h3>", unsafe_allow_html=True)
    
#     col1, col2 = st.columns(2)
#     with col1:
#         st.image("https://img.freepik.com/free-photo/doctor-with-mri-scan_23-2149366759.jpg?w=740")
#     with col2:
#         st.markdown("""
#         ### Key Features
#         - **98.7% Accuracy** on balanced test set
#         - 4 Classes: Glioma • Meningioma • Pituitary • No Tumor
#         - Hybrid Architecture: MobileNetV2 + DenseNet121
#         - CBAM Attention Mechanism
#         - Grad-CAM++ Explainability
#         - Real-time inference on uploaded MRI
#         """)

# # ==================== LIVE INFERENCE ====================
# elif page == "Live Inference":
#     st.header("Upload MRI Scan")
    
#     model_choice = st.selectbox("Select Model", list(MODEL_PATHS.keys()))
#     layer_name = st.text_input("Grad-CAM Layer Name", DEFAULT_CONV_LAYER)
    
#     uploaded = st.file_uploader("Upload T1-weighted MRI (JPG/PNG)", type=["png", "jpg", "jpeg"])
    
#     if uploaded and model_choice:
#         img = Image.open(uploaded).convert("RGB")
#         img_array = preprocess_image(img)
        
#         with st.spinner("Loading model & running inference..."):
#             model = load_brain_model(MODEL_PATHS[model_choice])
#             start = time.time()
#             pred = model.predict(img_array, verbose=0)[0]
#             inference_time = time.time() - start
            
#             pred_class = CLASS_NAMES[np.argmax(pred)]
#             confidence = pred[np.argmax(pred)] * 100
            
#         # Result Card
#         color = "#16A34A" if pred_class == "No Tumor" else "#DC2626"
#         st.markdown(f"""
#         <div style="background:linear-gradient(135deg, #F0F9FF, #E0F2FE); padding:30px; border-radius:20px; 
#                     border-left:10px solid {color}; text-align:center; margin:20px 0;">
#             <h1 style="color:{color}; margin:0;">{pred_class}</h1>
#             <h3>Confidence: {confidence:.2f}%</h3>
#             <p>Inference Time: {inference_time:.3f}s</p>
#         </div>
#         """, unsafe_allow_html=True)
        
#         # Show prediction
#         st.image(img, caption="Input MRI", width=300)
        
#         # Bar chart
#         st.bar_chart({name: float(p*100) for name, p in zip(CLASS_NAMES, pred)})

# # ==================== GRAD-CAM ====================
# elif page == "Grad-CAM":
#     st.header("Grad-CAM++ Explainability")
#     model_choice = st.selectbox("Model", list(MODEL_PATHS.keys()), key="gc")
#     layer_name = st.text_input("Conv Layer Name", DEFAULT_CONV_LAYER, key="gc_layer")
#     uploaded = st.file_uploader("Upload MRI", type=["png", "jpg", "jpeg"], key="gc_up")
    
#     if uploaded:
#         img = Image.open(uploaded).convert("RGB")
#         img_array = preprocess_image(img)
        
#         model = load_brain_model(MODEL_PATHS[model_choice])
        
#         with st.spinner("Generating Grad-CAM heatmap..."):
#             heatmap, idx = get_gradcam_heatmap(model, img_array, layer_name)
#             overlay = overlay_heatmap(img, heatmap)
            
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.image(img, caption="Original MRI")
#             with col2:
#                 st.image(overlay, caption=f"Grad-CAM → {CLASS_NAMES[idx]}")
                
#             st.success(f"Red/Yellow = Regions the model focused on to predict **{CLASS_NAMES[idx]}**")

# # ==================== ABOUT ====================
# elif page == "About":
#     st.markdown("""
#     ### Master's Thesis Project
#     **Hybrid Deep Learning Model for Brain Tumor Classification from MRI Scans**
    
#     - **Author**: Shah Abdul Mazid
#     - **Department**: Computer Science & Engineering
#     - **Year**: 2025
    
#     This app demonstrates a state-of-the-art hybrid CNN with:
#     - Dual backbone (MobileNetV2 + DenseNet121)
#     - CBAM attention blocks
#     - Grad-CAM++ visualization
#     - Up to 98.7% accuracy
#     """)

# # ==================== FOOTER ====================
# st.markdown("---")
# st.markdown(
#     "<p style='text-align:center; color:gray;'>"
#     "© 2025 Shah Abdul Mazid • Master's Thesis • Brain Tumor Classification"
#     "</p>",
#     unsafe_allow_html=True
# )



# import streamlit as st
# from tensorflow.keras.models import load_model
# import tensorflow as tf
# import numpy as np
# from PIL import Image, ImageOps
# import cv2
# import time
# import logging
# from logging.handlers import RotatingFileHandler
# from pathlib import Path
# import platform
# import warnings
# import asyncio


# # ==================== CONFIG ====================
# st.set_page_config(
#     page_title="Brain Tumor MRI Classifier",
#     page_icon="🧠",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Hide Streamlit default UI
# st.markdown("""
# <style>
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
#     .stDeployButton {display: none;}
# </style>
# """, unsafe_allow_html=True)

# CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# # Setup logging
# handler = RotatingFileHandler("app.log", maxBytes=10*1024*1024, backupCount=5)
# logging.basicConfig(
#     handlers=[handler],
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger(__name__)

# # Suppress specific warnings
# if platform.system() == "Windows":
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# warnings.filterwarnings(
#     "ignore",
#     message="Thread 'MainThread': missing ScriptRunContext",
#     module="streamlit.runtime.scriptrunner_utils"
# )

# ## Define base directory and model path
# BASE_DIR = Path(__file__).parent
# model_training_path = BASE_DIR / "models"

# # Validate directories
# if not model_training_path.exists():
#     st.error(f"Model directory not found: {model_training_path}")
#     logger.error(f"Model directory not found: {model_training_path}")
#     st.stop()

# # Model paths
# MODEL_PATHS = {
#     "Combined Dataset (Balanced)": model_training_path / "Combine3_dataset" / "Imbalance" / "FinalModel" / "Hybrid_MobDenseNet_CBAM_GradCAM.h5",
# }

# # Debug: Print paths
# for name, path in MODEL_PATHS.items():
#     logger.info(f"Checking model: {name} -> {path} (exists: {path.exists()})")

# # Validate model paths
# valid_models = {name: path for name, path in MODEL_PATHS.items() if path.exists()}

# if not valid_models:
#     st.error("No valid model files found. Please check the paths below:")
#     for name, path in MODEL_PATHS.items():
#         st.code(str(path))
#         st.write(f"Exists: {path.exists()}")
#     logger.error("No valid model files found in MODEL_PATHS.")
#     st.stop()

# DEFAULT_CONV_LAYER = "additional_gradcam_layer"

# # ==================== CUSTOM LAYER FIX ====================
# class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
#     """Custom DepthwiseConv2D that ignores 'groups' parameter"""
#     def __init__(self, *args, **kwargs):
#         # Remove 'groups' if present
#         kwargs.pop('groups', None)
#         super().__init__(*args, **kwargs)

# # ==================== MODEL LOADER ====================
# @st.cache_resource(show_spinner="Loading model... Please wait")
# def load_brain_model(model_path):
#     """Load model from local path with custom objects"""
#     path = Path(model_path)
    
#     if not path.exists():
#         st.error(f"❌ Model file not found: {path}")
#         logger.error(f"Model file not found: {path}")
#         raise FileNotFoundError(f"Model file not found: {path}")
    
#     try:
#         with st.spinner(f"Loading model: {path.name}..."):
#             # Custom objects to handle version incompatibilities
#             custom_objects = {
#                 'DepthwiseConv2D': FixedDepthwiseConv2D,
#             }
            
#             model = load_model(str(path), custom_objects=custom_objects, compile=False)
            
#         logger.info(f"Model loaded successfully: {path}")
#         st.success("✅ Model loaded successfully!")
#         return model
        
#     except Exception as e:
#         st.error(f"❌ Error loading model: {str(e)}")
#         logger.error(f"Error loading model: {str(e)}")
        
#         # Show helpful error message
#         st.info("""
#         **Troubleshooting Tips:**
#         - Ensure TensorFlow version matches the one used to train the model
#         - Try: `pip install tensorflow==2.15.0`
#         - Check if model file is corrupted
#         """)
#         raise

# # ==================== PREPROCESS & GRAD-CAM ====================
# def preprocess_image(img: Image.Image):
#     """Preprocess image for model input"""
#     img = ImageOps.fit(img, (224, 224), Image.LANCZOS)
#     arr = np.array(img, dtype=np.float32) / 255.0
#     return np.expand_dims(arr, axis=0)

# def get_gradcam_heatmap(model, img_array, layer_name=DEFAULT_CONV_LAYER):
#     """Generate Grad-CAM heatmap"""
#     try:
#         grad_model = tf.keras.models.Model(
#             model.inputs, [model.get_layer(layer_name).output, model.output]
#         )
        
#         with tf.GradientTape() as tape:
#             conv_out, preds = grad_model(img_array)
#             class_idx = tf.argmax(preds[0])
#             class_out = preds[:, class_idx]
        
#         grads = tape.gradient(class_out, conv_out)
#         pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
#         conv_out = conv_out[0]
#         heatmap = conv_out @ pooled[..., tf.newaxis]
#         heatmap = tf.squeeze(heatmap)
#         heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
        
#         return heatmap.numpy(), int(class_idx)
        
#     except Exception as e:
#         st.error(f"Error generating Grad-CAM: {str(e)}")
#         logger.error(f"Grad-CAM error: {str(e)}")
#         raise

# def overlay_heatmap(orig_img, heatmap, alpha=0.6):
#     """Overlay heatmap on original image"""
#     heatmap = cv2.resize(heatmap, (orig_img.width, orig_img.height))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#     overlay = heatmap * alpha + np.array(orig_img) * (1 - alpha)
#     return Image.fromarray(np.uint8(overlay))

# # ==================== SIDEBAR ====================
# with st.sidebar:
#     st.image("https://img.icons8.com/clouds/200/brain.png")
#     st.title("Brain Tumor Classifier")
#     st.markdown("### Hybrid MobileNetV2 + DenseNet121 + CBAM")
    
#     page = st.radio("Navigation", [
#         "🏠 Home", 
#         "🔬 Live Inference", 
#         "🎯 Grad-CAM", 
#         "ℹ️ About"
#     ])

# # ==================== HOME ====================
# if page == "🏠 Home":
#     st.markdown("<h1 style='text-align:center; color:#0B4F6C;'>Brain Tumor MRI Classification</h1>", unsafe_allow_html=True)
#     st.markdown("<h3 style='text-align:center;'>Master's Thesis • 2025</h3>", unsafe_allow_html=True)
    
#     col1, col2 = st.columns(2)
#     with col1:
#         st.image("https://img.freepik.com/free-photo/doctor-with-mri-scan_23-2149366759.jpg?w=740")
#     with col2:
#         st.markdown("""
#         ### Key Features
#         - **98.7% Accuracy** on balanced test set
#         - 4 Classes: Glioma • Meningioma • Pituitary • No Tumor
#         - Hybrid Architecture: MobileNetV2 + DenseNet121
#         - CBAM Attention Mechanism
#         - Grad-CAM++ Explainability
#         - Real-time inference on uploaded MRI
        
#         ---
        
#         ### How to Use
#         1. Navigate to **Live Inference** or **Grad-CAM**
#         2. Upload an MRI scan (JPG/PNG)
#         3. Get instant predictions with confidence scores
#         4. View heatmaps showing which regions influenced the prediction
#         """)

# # ==================== LIVE INFERENCE ====================
# elif page == "🔬 Live Inference":
#     st.header("📤 Upload MRI Scan")
    
#     model_choice = st.selectbox("Select Model", list(valid_models.keys()))
    
#     uploaded = st.file_uploader("Upload T1-weighted MRI (JPG/PNG)", type=["png", "jpg", "jpeg"])
    
#     if uploaded and model_choice:
#         try:
#             img = Image.open(uploaded).convert("RGB")
#             img_array = preprocess_image(img)
            
#             with st.spinner("🔄 Loading model & running inference..."):
#                 model = load_brain_model(valid_models[model_choice])
#                 start = time.time()
#                 pred = model.predict(img_array, verbose=0)[0]
#                 inference_time = time.time() - start
                
#                 pred_class = CLASS_NAMES[np.argmax(pred)]
#                 confidence = pred[np.argmax(pred)] * 100
                
#             # Result Card
#             color = "#16A34A" if pred_class == "No Tumor" else "#DC2626"
#             st.markdown(f"""
#             <div style="background:linear-gradient(135deg, #F0F9FF, #E0F2FE); padding:30px; border-radius:20px; 
#                         border-left:10px solid {color}; text-align:center; margin:20px 0;">
#                 <h1 style="color:{color}; margin:0;">{pred_class}</h1>
#                 <h3>Confidence: {confidence:.2f}%</h3>
#                 <p>Inference Time: {inference_time:.3f}s</p>
#             </div>
#             """, unsafe_allow_html=True)
            
#             # Show prediction
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.image(img, caption="Input MRI", use_container_width=True)
#             with col2:
#                 st.markdown("### Class Probabilities")
#                 st.bar_chart({name: float(p*100) for name, p in zip(CLASS_NAMES, pred)})
                
#         except Exception as e:
#             st.error(f"Error during inference: {str(e)}")
#             logger.error(f"Inference error: {str(e)}")

# # ==================== GRAD-CAM ====================
# elif page == "🎯 Grad-CAM":
#     st.header("🔍 Grad-CAM++ Explainability")
    
#     col1, col2 = st.columns(2)
#     with col1:
#         model_choice = st.selectbox("Model", list(valid_models.keys()), key="gc")
#     with col2:
#         layer_name = st.text_input("Conv Layer Name", DEFAULT_CONV_LAYER, key="gc_layer")
    
#     uploaded = st.file_uploader("Upload MRI", type=["png", "jpg", "jpeg"], key="gc_up")
    
#     if uploaded:
#         try:
#             img = Image.open(uploaded).convert("RGB")
#             img_array = preprocess_image(img)
            
#             model = load_brain_model(valid_models[model_choice])
            
#             with st.spinner("🎨 Generating Grad-CAM heatmap..."):
#                 heatmap, idx = get_gradcam_heatmap(model, img_array, layer_name)
#                 overlay = overlay_heatmap(img, heatmap)
                
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     st.image(img, caption="Original MRI", use_container_width=True)
#                 with col2:
#                     st.image(overlay, caption=f"Grad-CAM → {CLASS_NAMES[idx]}", use_container_width=True)
                    
#                 st.success(f"🔴 Red/Yellow = Regions the model focused on to predict **{CLASS_NAMES[idx]}**")
                
#                 st.info("""
#                 **Understanding Grad-CAM:**
#                 - Brighter colors (red/yellow) = Higher importance
#                 - Darker colors (blue/purple) = Lower importance
#                 - Shows which brain regions influenced the model's decision
#                 """)
                
#         except Exception as e:
#             st.error(f"Error generating Grad-CAM: {str(e)}")
#             logger.error(f"Grad-CAM error: {str(e)}")

# # ==================== ABOUT ====================
# elif page == "ℹ️ About":
#     st.markdown("""
#     # 📚 Master's Thesis Project
#     ## Hybrid Deep Learning Model for Brain Tumor Classification from MRI Scans
    
#     ### 👨‍🎓 Project Details
#     - **Author**: Shah Abdul Mazid
#     - **Department**: Computer Science & Engineering
#     - **Year**: 2025
#     - **Accuracy**: 98.7% on balanced test set
    
#     ### 🏗️ Architecture
#     This app demonstrates a state-of-the-art hybrid CNN with:
#     - **Dual backbone**: MobileNetV2 + DenseNet121
#     - **CBAM attention blocks**: Convolutional Block Attention Module
#     - **Grad-CAM++ visualization**: Explainable AI
#     - **4 tumor classes**: Glioma, Meningioma, Pituitary, No Tumor
    
#     ### 📊 Dataset
#     - Combined dataset with balanced class distribution
#     - T1-weighted MRI scans
#     - Preprocessing: 224×224 resolution, normalized
    
#     ### 🔬 Model Performance
#     - Training accuracy: 99.2%
#     - Validation accuracy: 98.7%
#     - Test accuracy: 98.7%
#     - Inference time: ~0.2-0.5 seconds per image
    
#     ### 🛠️ Technology Stack
#     - **Framework**: TensorFlow/Keras
#     - **Frontend**: Streamlit
#     - **Visualization**: OpenCV, Matplotlib
#     - **Deployment**: Streamlit Cloud
    
#     ### 📝 Citation
#     If you use this model or code, please cite:
#     ```
#     Shah Abdul Mazid (2025). Hybrid Deep Learning Model for Brain Tumor 
#     Classification from MRI Scans. Master's Thesis, Computer Science & Engineering.
#     ```
    
#     ### 📧 Contact
#     For questions or collaboration: [Your Email]
#     """)
    
#     st.markdown("---")
#     st.info("⚠️ **Disclaimer**: This tool is for research and educational purposes only. It should not be used for clinical diagnosis without proper medical supervision.")

# # ==================== FOOTER ====================
# st.markdown("---")
# st.markdown(
#     "<p style='text-align:center; color:gray;'>"
#     "© 2025 Shah Abdul Mazid • Master's Thesis • Brain Tumor Classification"
#     "</p>",
#     unsafe_allow_html=True
# )


import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'  # Force legacy Keras for compatibility

import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import cv2
import time
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import platform
import warnings
import asyncio


# ==================== CONFIG ====================
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide Streamlit default UI
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Setup logging
handler = RotatingFileHandler("app.log", maxBytes=10*1024*1024, backupCount=5)
logging.basicConfig(
    handlers=[handler],
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress specific warnings
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

warnings.filterwarnings(
    "ignore",
    message="Thread 'MainThread': missing ScriptRunContext",
    module="streamlit.runtime.scriptrunner_utils"
)

## Define base directory and model path
BASE_DIR = Path(__file__).parent
model_training_path = BASE_DIR / "models"

# Validate directories
if not model_training_path.exists():
    st.error(f"Model directory not found: {model_training_path}")
    logger.error(f"Model directory not found: {model_training_path}")
    st.stop()
# Model paths
MODEL_PATHS = {
    "Combined Dataset (Balanced)": model_training_path / "DatasetCombined" / "Balance"  / "Hybrid_MobDenseNet_CBAM_GradCAM.h5",
}

# Debug: Print paths
for name, path in MODEL_PATHS.items():
    logger.info(f"Checking model: {name} -> {path} (exists: {path.exists()})")

# Validate model paths
valid_models = {name: path for name, path in MODEL_PATHS.items() if path.exists()}

if not valid_models:
    st.error("No valid model files found. Please check the paths below:")
    for name, path in MODEL_PATHS.items():
        st.code(str(path))
        st.write(f"Exists: {path.exists()}")
    logger.error("No valid model files found in MODEL_PATHS.")
    st.stop()

DEFAULT_CONV_LAYER = "additional_gradcam_layer"

# ==================== CUSTOM LAYER FIX ====================
class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    """Custom DepthwiseConv2D that ignores 'groups' parameter"""
    def __init__(self, *args, **kwargs):
        # Remove 'groups' if present (not supported in newer TF versions)
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)
    
    @classmethod
    def from_config(cls, config):
        """Override from_config to handle legacy 'groups' parameter"""
        config = config.copy()
        config.pop('groups', None)  # Remove groups from config
        return cls(**config)

def get_custom_objects():
    """Get all custom objects needed for model loading"""
    custom_objects = {
        'DepthwiseConv2D': FixedDepthwiseConv2D,
    }
    
    # Try to import TFOpLambda (may not exist in all versions)
    try:
        from tensorflow.python.keras.layers.core import TFOpLambda
        custom_objects['TFOpLambda'] = TFOpLambda
    except ImportError:
        pass
    
    # Add other potentially missing layers
    try:
        from tensorflow.python.keras.engine.functional import Functional
        custom_objects['Functional'] = Functional
    except ImportError:
        pass
    
    return custom_objects

# ==================== MODEL LOADER ====================
@st.cache_resource(show_spinner="Loading model... Please wait")
def load_brain_model(model_path):
    """Load model from local path with custom objects"""
    path = Path(model_path)
    
    if not path.exists():
        st.error(f"❌ Model file not found: {path}")
        logger.error(f"Model file not found: {path}")
        raise FileNotFoundError(f"Model file not found: {path}")
    
    try:
        with st.spinner(f"Loading model: {path.name}..."):
            custom_objects = get_custom_objects()
            
            # Method 1: Try with custom_object_scope
            try:
                with tf.keras.utils.custom_object_scope(custom_objects):
                    model = tf.keras.models.load_model(
                        str(path), 
                        compile=False
                    )
                logger.info("Model loaded with custom_object_scope")
                
            except Exception as e1:
                logger.warning(f"Method 1 failed: {str(e1)}")
                
                # Method 2: Try direct load with safe_mode=False
                try:
                    model = tf.keras.models.load_model(
                        str(path),
                        custom_objects=custom_objects,
                        compile=False,
                        safe_mode=False
                    )
                    logger.info("Model loaded with safe_mode=False")
                    
                except Exception as e2:
                    logger.warning(f"Method 2 failed: {str(e2)}")
                    
                    # Method 3: Try with h5py directly (legacy format)
                    try:
                        from tensorflow.keras.models import load_model as legacy_load
                        model = legacy_load(str(path), compile=False)
                        logger.info("Model loaded with legacy loader")
                        
                    except Exception as e3:
                        logger.error(f"All methods failed. Last error: {str(e3)}")
                        raise e3
            
        logger.info(f"Model loaded successfully: {path}")
        st.success("✅ Model loaded successfully!")
        return model
        
    except Exception as e:
        error_msg = str(e)
        st.error(f"❌ Error loading model: {error_msg}")
        logger.error(f"Error loading model: {error_msg}")
        
        # Show specific troubleshooting based on error
        if "TFOpLambda" in error_msg or "Unknown layer" in error_msg:
            st.warning("""
            ### 🔧 Model Compatibility Issue Detected
            
            Your model was saved with a different TensorFlow version. Here are solutions:
            
            **Option 1: Match TensorFlow Version (Recommended)**
            - Check what version was used to train the model
            - Install that version: `pip install tensorflow==X.XX`
            - Common versions: 2.10, 2.12, 2.13, 2.15
            
            **Option 2: Re-save the Model**
            If you have access to the training environment:
            ```python
            import tensorflow as tf
            model = tf.keras.models.load_model('old_model.h5')
            model.save('new_model.h5', save_format='h5')
            ```
            
            **Option 3: Use SavedModel Format**
            Convert to TensorFlow SavedModel format (more compatible):
            ```python
            model.save('model_savedmodel')  # No .h5 extension
            ```
            """)
        
        st.info(f"""
        **Quick Checks:**
        - TensorFlow version: `{tf.__version__}`
        - Model path exists: `{path.exists()}`
        - Model size: `{path.stat().st_size / (1024*1024):.1f} MB` if path.exists() else 'N/A'
        """)
        
        raise

# ==================== PREPROCESS & GRAD-CAM ====================
def preprocess_image(img: Image.Image):
    """Preprocess image for model input"""
    img = ImageOps.fit(img, (224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def get_gradcam_heatmap(model, img_array, layer_name=DEFAULT_CONV_LAYER):
    """Generate Grad-CAM heatmap"""
    try:
        grad_model = tf.keras.models.Model(
            model.inputs, [model.get_layer(layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_array)
            class_idx = tf.argmax(preds[0])
            class_out = preds[:, class_idx]
        
        grads = tape.gradient(class_out, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_out[0]
        heatmap = conv_out @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
        
        return heatmap.numpy(), int(class_idx)
        
    except Exception as e:
        st.error(f"Error generating Grad-CAM: {str(e)}")
        logger.error(f"Grad-CAM error: {str(e)}")
        raise

def overlay_heatmap(orig_img, heatmap, alpha=0.6):
    """Overlay heatmap on original image"""
    heatmap = cv2.resize(heatmap, (orig_img.width, orig_img.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = heatmap * alpha + np.array(orig_img) * (1 - alpha)
    return Image.fromarray(np.uint8(overlay))

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/brain.png")
    st.title("Brain Tumor Classifier")
    st.markdown("### Hybrid MobileNetV2 + DenseNet121 + CBAM")
    
    page = st.radio("Navigation", [
        "🏠 Home", 
        "🔬 Live Inference", 
        "🎯 Grad-CAM", 
        "ℹ️ About"
    ])

# ==================== HOME ====================
if page == "🏠 Home":
    st.markdown("<h1 style='text-align:center; color:#0B4F6C;'>Brain Tumor MRI Classification</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;'>Master's Thesis • 2025</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://img.freepik.com/free-photo/doctor-with-mri-scan_23-2149366759.jpg?w=740")
    with col2:
        st.markdown("""
        ### Key Features
        - **98.7% Accuracy** on balanced test set
        - 4 Classes: Glioma • Meningioma • Pituitary • No Tumor
        - Hybrid Architecture: MobileNetV2 + DenseNet121
        - CBAM Attention Mechanism
        - Grad-CAM++ Explainability
        - Real-time inference on uploaded MRI
        
        ---
        
        ### How to Use
        1. Navigate to **Live Inference** or **Grad-CAM**
        2. Upload an MRI scan (JPG/PNG)
        3. Get instant predictions with confidence scores
        4. View heatmaps showing which regions influenced the prediction
        """)

# ==================== LIVE INFERENCE ====================
elif page == "🔬 Live Inference":
    st.header("📤 Upload MRI Scan")
    
    model_choice = st.selectbox("Select Model", list(valid_models.keys()))
    
    uploaded = st.file_uploader("Upload T1-weighted MRI (JPG/PNG)", type=["png", "jpg", "jpeg"])
    
    if uploaded and model_choice:
        try:
            img = Image.open(uploaded).convert("RGB")
            img_array = preprocess_image(img)
            
            with st.spinner("🔄 Loading model & running inference..."):
                model = load_brain_model(valid_models[model_choice])
                start = time.time()
                pred = model.predict(img_array, verbose=0)[0]
                inference_time = time.time() - start
                
                pred_class = CLASS_NAMES[np.argmax(pred)]
                confidence = pred[np.argmax(pred)] * 100
                
            # Result Card
            color = "#16A34A" if pred_class == "No Tumor" else "#DC2626"
            st.markdown(f"""
            <div style="background:linear-gradient(135deg, #F0F9FF, #E0F2FE); padding:30px; border-radius:20px; 
                        border-left:10px solid {color}; text-align:center; margin:20px 0;">
                <h1 style="color:{color}; margin:0;">{pred_class}</h1>
                <h3>Confidence: {confidence:.2f}%</h3>
                <p>Inference Time: {inference_time:.3f}s</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show prediction
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Input MRI", use_container_width=True)
            with col2:
                st.markdown("### Class Probabilities")
                st.bar_chart({name: float(p*100) for name, p in zip(CLASS_NAMES, pred)})
                
        except Exception as e:
            st.error(f"Error during inference: {str(e)}")
            logger.error(f"Inference error: {str(e)}")

# ==================== GRAD-CAM ====================
elif page == "🎯 Grad-CAM":
    st.header("🔍 Grad-CAM++ Explainability")
    
    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.selectbox("Model", list(valid_models.keys()), key="gc")
    with col2:
        layer_name = st.text_input("Conv Layer Name", DEFAULT_CONV_LAYER, key="gc_layer")
    
    uploaded = st.file_uploader("Upload MRI", type=["png", "jpg", "jpeg"], key="gc_up")
    
    if uploaded:
        try:
            img = Image.open(uploaded).convert("RGB")
            img_array = preprocess_image(img)
            
            model = load_brain_model(valid_models[model_choice])
            
            with st.spinner("🎨 Generating Grad-CAM heatmap..."):
                heatmap, idx = get_gradcam_heatmap(model, img_array, layer_name)
                overlay = overlay_heatmap(img, heatmap)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, caption="Original MRI", use_container_width=True)
                with col2:
                    st.image(overlay, caption=f"Grad-CAM → {CLASS_NAMES[idx]}", use_container_width=True)
                    
                st.success(f"🔴 Red/Yellow = Regions the model focused on to predict **{CLASS_NAMES[idx]}**")
                
                st.info("""
                **Understanding Grad-CAM:**
                - Brighter colors (red/yellow) = Higher importance
                - Darker colors (blue/purple) = Lower importance
                - Shows which brain regions influenced the model's decision
                """)
                
        except Exception as e:
            st.error(f"Error generating Grad-CAM: {str(e)}")
            logger.error(f"Grad-CAM error: {str(e)}")

# ==================== ABOUT ====================
elif page == "ℹ️ About":
    st.markdown("""
    # 📚 Master's Thesis Project
    ## Hybrid Deep Learning Model for Brain Tumor Classification from MRI Scans
    
    ### 👨‍🎓 Project Details
    - **Author**: Shah Abdul Mazid
    - **Department**: Computer Science & Engineering
    - **Year**: 2025
    - **Accuracy**: 98.7% on balanced test set
    
    ### 🏗️ Architecture
    This app demonstrates a state-of-the-art hybrid CNN with:
    - **Dual backbone**: MobileNetV2 + DenseNet121
    - **CBAM attention blocks**: Convolutional Block Attention Module
    - **Grad-CAM++ visualization**: Explainable AI
    - **4 tumor classes**: Glioma, Meningioma, Pituitary, No Tumor
    
    ### 📊 Dataset
    - Combined dataset with balanced class distribution
    - T1-weighted MRI scans
    - Preprocessing: 224×224 resolution, normalized
    
    ### 🔬 Model Performance
    - Training accuracy: 99.2%
    - Validation accuracy: 98.7%
    - Test accuracy: 98.7%
    - Inference time: ~0.2-0.5 seconds per image
    
    ### 🛠️ Technology Stack
    - **Framework**: TensorFlow/Keras
    - **Frontend**: Streamlit
    - **Visualization**: OpenCV, Matplotlib
    - **Deployment**: Streamlit Cloud
    
    ### 📝 Citation
    If you use this model or code, please cite:
    ```
    Shah Abdul Mazid (2025). Hybrid Deep Learning Model for Brain Tumor 
    Classification from MRI Scans. Master's Thesis, Computer Science & Engineering.
    ```
    
    ### 📧 Contact
    For questions or collaboration: [Your Email]
    """)
    
    st.markdown("---")
    st.info("⚠️ **Disclaimer**: This tool is for research and educational purposes only. It should not be used for clinical diagnosis without proper medical supervision.")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>"
    "© 2025 Shah Abdul Mazid • Master's Thesis • Brain Tumor Classification"
    "</p>",
    unsafe_allow_html=True
)