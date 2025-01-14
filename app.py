import os
import logging
from ultralytics import YOLO
import streamlit as st
from PIL import ImageOps, Image
import torch
import time
import pandas as pd
import settings

# Handle OpenMP library duplication issues
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Streamlit page configuration
st.set_page_config(
    page_title="Object Detection using YOLO",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define directories and logging
script_directory = os.path.dirname(os.path.abspath(__file__))
log_directory = os.path.join(script_directory, "log")
log_filename = os.path.join(log_directory, 'debug.log')

# Ensure log directory exists
os.makedirs(log_directory, exist_ok=True)

# Clear old logs
with open(log_filename, 'w') as log_file:
    pass

# Configure logging
logging.basicConfig(filename=log_filename, level=logging.DEBUG)

# Load local CSS
def load_local_css(file_name):
    try:
        with open(os.path.join(script_directory, file_name), "r") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file '{file_name}' not found.")
        logging.error(f"CSS file '{file_name}' not found.")

load_local_css("style.css")

# Initialize YOLO models
yolo_models = {'yolov8n': None, 'yolov11n': None}
MODEL_DIR = settings.MODEL_DIR

# Load models
for model_key in yolo_models.keys():
    model_path = os.path.join(MODEL_DIR, f"{model_key}.pt")
    try:
        yolo_models[model_key] = YOLO(model_path).to('cuda' if torch.cuda.is_available() else 'cpu')
    except Exception as ex:
        st.error(f"Unable to load model {model_key}: {model_path}")
        logging.error(f"Error loading model {model_key}: {model_path}\n{ex}")

# Header
st.markdown('<p class="title-center">Multiclass Coffee Bean Detection System using Deep Learning</p>', unsafe_allow_html=True)

# Sidebar
confidence = st.sidebar.slider("Model Confidence Level", 0.25, 1.0, 0.4, 0.01)
source_select = st.sidebar.selectbox("Select Detection Object", settings.SOURCES_LIST)
source_img = st.sidebar.file_uploader("Select Image File", type=("jpg", "jpeg", "png", "bmp", "webp"))

# Default image path
default_image_path = settings.DEFAULT_IMAGE

# Detection processing
if source_img and st.sidebar.button("Process Detection"):
    st.subheader("Detection Process")
    col1, col2, col3 = st.columns(3)

    # Column 1: Display image
    with col1:
        try:
            image = Image.open(source_img)
            image = ImageOps.exif_transpose(image)
            image = image.resize((640, int(640 / image.width * image.height)))
            st.image(image, caption="Uploaded Image", use_container_width=True)
        except Exception as ex:
            st.error(f"Error processing the uploaded image: {ex}")
            logging.error(f"Error processing the uploaded image: {ex}")

    # YOLO Detection Results
    data = []
    for model_key, model in yolo_models.items():
        with col2 if model_key == 'yolov8n' else col3:
            try:
                start_time = time.time()
                results = model.predict(image, conf=confidence)
                end_time = time.time()

                plotted_image = results[0].plot()[:, :, ::-1]
                st.image(plotted_image, caption=f"{model_key} Detection Results", use_container_width=True)

                detected_boxes = [box for box in results[0].boxes if box.conf.item() > confidence]
                class_names = [settings.CLASS_NAMES[int(box.cls.item())] for box in detected_boxes]
                confidences = [float(box.conf.item()) for box in detected_boxes]

                data.append({
                    "Model": model_key,
                    "Detected Classes": ', '.join(set(class_names)) if class_names else "None",
                    "Average Confidence": f"{sum(confidences) / len(confidences):.2f}" if confidences else "-",
                    "Detection Time": f"{end_time - start_time:.3f} seconds"
                })
            except Exception as ex:
                st.error(f"Error processing with {model_key}: {ex}")
                logging.error(f"Error processing with {model_key}: {ex}")

    # Display analysis table
    st.subheader("Statistical Analysis of Detection Results")
    st.table(pd.DataFrame(data))
else:
    st.warning("Select an image to start detection.")

# Sidebar: GPU availability
with st.sidebar:
    gpu_status = torch.cuda.is_available()
    st.write(f"GPU Available: {torch.cuda.get_device_name(0)}" if gpu_status else "No GPU available.")
    logging.debug(f"GPU status: {'Available' if gpu_status else 'Not available'}")
