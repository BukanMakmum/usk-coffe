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


