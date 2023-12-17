import streamlit as lit
import numpy as np

BUTTON_LABEL="Go ahead and predict the result using the model!"
UPLOAD_STR="Upload your image here."
CAMERA_UPLOAD_STR = "Or, take a photo instead!"

lit.title("Melanoma Classifier")

uploaded_img = lit.file_uploader(UPLOAD_STR)

cam_img = lit.camera_input(CAMERA_UPLOAD_STR)

if uploaded_img is not None:
    uploaded_img = uploaded_img.getvalue()
    
if cam_img is not None and uploaded_img is not None:
    lit.subheader("Choose between the camera or the file input", divider="rainbow")
    lit.button("Uploaded image!")
    lit.button("Camera input, please.")

if uploaded_img is not None or cam_img is not None:
    final_val = uploaded_img
    lit.button(BUTTON_LABEL)