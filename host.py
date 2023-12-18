import streamlit as lit
import numpy as np

BUTTON_LABEL="Go ahead and predict the result using the model!"
CAMERA_CHOICE_STR = "Camera input, please."
CAMERA_UPLOAD_STR = "Or, take a photo instead!"
CHOICE_WARN_STR = "Choose between the camera or the file input. The option you\
        choose first will be the input that the model predicts on."
RAINBOW="rainbow"
SUCCESS_MSG = "Congratulations! No melanoma detected."
TITLE = "Melanoma Classifier"
UPLOAD_CHOICE_STR = "Uploaded image!"
UPLOAD_STR="Upload your image here."
WARNING_MSG = "Uh oh, looks like this tested positive for melanoma."

MODEL = None # model.pkl TODO
def _predict(input):
    
    if input:
        lit.warning(WARNING_MSG)

    else:
        lit.success(SUCCESS_MSG)



lit.title(TITLE)

uploaded_img = lit.file_uploader(UPLOAD_STR)

cam_img = lit.camera_input(CAMERA_UPLOAD_STR)

if uploaded_img is not None:
    uploaded_img = uploaded_img.getvalue()

# TODO: input is a stream of bytes, have to pass in as ndarray into _predict.
if cam_img is not None and uploaded_img is not None:
    lit.subheader(CHOICE_WARN_STR, divider=RAINBOW)
    lit.button(UPLOAD_CHOICE_STR, on_click=_predict, args=(cam_img))
    lit.button(CAMERA_CHOICE_STR,on_click=_predict, args=(uploaded_img))

elif cam_img is not None or uploaded_img is not None:
    if uploaded_img is not None:
        final_img = uploaded_img
        
    if cam_img is not None:
        final_img = cam_img
    
    lit.button(BUTTON_LABEL,on_click=_predict, args=(final_img))