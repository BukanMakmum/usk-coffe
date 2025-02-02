import os

# Determine the script directory
file_path = os.path.abspath(__file__)
root_path = os.path.dirname(file_path)

# Define directories for images and models
IMAGES_DIR = os.path.join(root_path, 'images')
MODEL_DIR = os.path.join(root_path, 'model')

# Path for default images
DEFAULT_IMAGE = os.path.join(IMAGES_DIR, 'Default_Image.jpg')  # Ensure this default image exists in the folder
DEFAULT_DETECT_IMAGE = os.path.join(IMAGES_DIR, 'Detected_Image.jpg')  # Image for detection output

# Store paths for other sources
SOURCES_LIST = ['Image']
IMAGE = 'Image'

import os
import logging
from ultralytics import YOLO
import streamlit as st
from PIL import ImageOps, Image
import torch
import time
import pandas as pd  # Import pandas for creating tables
import settings

# Handle OpenMP library duplication issues on certain systems
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set Streamlit page configuration
st.set_page_config(
    page_title="Object Detection using YOLO",
    page_icon="💻",
    layout="wide",  # Use wide layout
    initial_sidebar_state="expanded"
)

# Define script directory
script_directory = os.path.dirname(os.path.abspath(__file__))

# Create log directory if it does not exist
log_directory = "d:/GitHub/Mini-Experiment/log"
log_filename = os.path.join(log_directory, 'debug.log')

# Clear old logs if they exist
if os.path.exists(log_filename):
    with open(log_filename, 'w'):  # Write to the log file to clear its contents
        pass

# Create log directory if it does not exist
os.makedirs(log_directory, exist_ok=True)

# Configure logging with the specified log file
logging.basicConfig(filename=log_filename, level=logging.DEBUG)

# Load local CSS for page styling
def local_css(file_name):
    with open(os.path.join(os.path.dirname(__file__), file_name), "r") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# Initialize YOLO models
yolo_models = {'yolov8n': None, 'yolov11n': None}

# Define model directory from settings.py
MODEL_DIR = settings.MODEL_DIR

# Build model file paths for both YOLO models
for model_key in yolo_models.keys():
    model_path = os.path.join(MODEL_DIR, f"{model_key}.pt")
    try:
        yolo_models[model_key] = YOLO(model_path).to('cuda' if torch.cuda.is_available() else 'cpu')
    except Exception as ex:
        st.error(f"Unable to load model {model_key}. Check the specified path: {model_path}")
        st.error(str(ex))
        logging.error(f"Unable to load model {model_key}. Check the specified path: {model_path}")
        logging.error(str(ex))

# Display header with formatted text
st.markdown(f'<p class="title-center">Multiclass Coffee Bean Detection System using Deep Learning</p>', unsafe_allow_html=True)

# Sidebar for confidence level settings and input source selection
confidence = st.sidebar.slider("Model Confidence Level", 0.25, 1.0, 0.4, 0.01)
source_select = st.sidebar.selectbox(f"Select Detection Object", settings.SOURCES_LIST)

source_img = None
boxes = {model_key: [] for model_key in yolo_models.keys()}

# Flag to mark if the process button has been clicked
process_button_clicked = False

# If image source is selected
if source_select == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Select Image File", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

# Define default image path from settings.py
default_image_path = settings.DEFAULT_IMAGE  # Use path from settings.py

# Process object detection if the button is clicked
if source_img is not None and st.sidebar.button("Process Detection"):
    process_button_clicked = True  # Mark that the button has been clicked

    # Object Detection Process
    st.subheader("Detection Process")

    # Create 3 columns
    col1, col2, col3 = st.columns(3)  # Create 3 columns

    # Column 1: Display default or uploaded image
    with col1:
        if not source_img:  # If no image is uploaded, display the default image
            try:
                # Load default image file
                default_image = Image.open(default_image_path)
                default_image = ImageOps.exif_transpose(default_image)  # Ensure correct image orientation
                st.image(default_image, caption="Default Image", use_container_width=True)
            except FileNotFoundError:
                # Handle error if default image file is not found
                st.error("Default image file not found.")
                logging.error("Default image file not found.")
            except Exception as ex:
                # Handle general errors when opening the image
                st.error(f"Error opening default image: {str(ex)}")
                logging.error(f"Error opening default image: {str(ex)}")
        else:  # If an image is uploaded, display the uploaded image
            uploaded_image = Image.open(source_img)
            uploaded_image = ImageOps.exif_transpose(uploaded_image)
            uploaded_image = uploaded_image.resize((640, int(640 / uploaded_image.width * uploaded_image.height)))
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    # Column 2: Display YOLOv8 detection results
    with col2:
        start_time_yolov8 = time.time()
        res_yolov8 = yolo_models['yolov8n'].predict(uploaded_image, conf=confidence)
        end_time_yolov8 = time.time()
        boxes['yolov8n'] = res_yolov8[0].boxes
        res_yolov8_plotted = res_yolov8[0].plot()[:, :, ::-1]
        st.image(res_yolov8_plotted, caption='YOLOv8 Detection Results', use_container_width=True)
        detection_time_yolov8 = end_time_yolov8 - start_time_yolov8

    # Column 3: Display YOLOv11 detection results
    with col3:
        start_time_yolov11 = time.time()
        res_yolov11 = yolo_models['yolov11n'].predict(uploaded_image, conf=confidence)
        end_time_yolov11 = time.time()
        boxes['yolov11n'] = res_yolov11[0].boxes
        res_yolov11_plotted = res_yolov11[0].plot()[:, :, ::-1]
        st.image(res_yolov11_plotted, caption='YOLOv11 Detection Results', use_container_width=True)
        detection_time_yolov11 = end_time_yolov11 - start_time_yolov11

    # Statistical Analysis of Detection Results
    st.subheader("Statistical Analysis of Detection Results")

    # Create analysis in table form
    data = []

    # Class dictionary with corresponding class names
    class_names = {
        0: "defect",
        1: "longberry",
        2: "peaberry",
        3: "premium"
    }

    # Analysis for YOLOv8
    if len(boxes['yolov8n']) > 0:
        # Access detection box coordinates for object counting
        detected_objects_yolov8 = [box for box in boxes['yolov8n'] if box.conf.item() > confidence]
        categories_yolov8 = [int(box.cls.item()) for box in detected_objects_yolov8]
        class_names_yolov8 = [class_names[cls] for cls in categories_yolov8]  # Replace ID with class names
        confidences_yolov8 = [float(box.conf.item()) for box in detected_objects_yolov8]
        data.append({
            "Model": "YOLOv8",
            "Detected Class Names": ', '.join(map(str, set(class_names_yolov8))),  # Display class names
            "Average Confidence Level": f"{sum(confidences_yolov8) / len(confidences_yolov8):.2f}",
            "Detection Time": f"{detection_time_yolov8:.3f} seconds"
        })
    else:
        data.append({
            "Model": "YOLOv8",
            "Detected Class Names": "None",
            "Average Confidence Level": "-",
            "Detection Time": "-"
        })

    # Analysis for YOLOv11
    if len(boxes['yolov11n']) > 0:
        # Access detection box coordinates for object counting
        detected_objects_yolov11 = [box for box in boxes['yolov11n'] if box.conf.item() > confidence]
        categories_yolov11 = [int(box.cls.item()) for box in detected_objects_yolov11]
        class_names_yolov11 = [class_names[cls] for cls in categories_yolov11]  # Replace ID with class names
        confidences_yolov11 = [float(box.conf.item()) for box in detected_objects_yolov11]
        data.append({
            "Model": "YOLOv11",
            "Detected Class Names": ', '.join(map(str, set(class_names_yolov11))),  # Display class names
            "Average Confidence Level": f"{sum(confidences_yolov11) / len(confidences_yolov11):.2f}",
            "Detection Time": f"{detection_time_yolov11:.3f} seconds"
        })
    else:
        data.append({
            "Model": "YOLOv11",
            "Detected Class Names": "None",
            "Average Confidence Level": "-",
            "Detection Time": "-"
        })

    # Create pandas dataframe from analysis data
    df = pd.DataFrame(data)

    # Display analysis table
    st.table(df)

# Check model status
else:
    st.warning("Select an image first to start detection.")

# Check if GPU is available
with st.sidebar:
    if torch.cuda.is_available():
        logging.debug(f'GPU found: {torch.cuda.get_device_name(0)}')
        st.write(f'Available GPU: {torch.cuda.get_device_name(0)}')
    else:
        logging.debug('No GPU available.')
        st.write('No GPU available for use.')
