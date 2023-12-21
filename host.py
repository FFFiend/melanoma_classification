"""
Frontend logic for streamlit.
"""
import numpy as np
import pickle # TODO 
import streamlit as lit
from io import BytesIO
from PIL import Image

BUTTON_LABEL="Go ahead and predict the result using the model!"
CAMERA_CHOICE_STR = "Camera input, please."
CAMERA_UPLOAD_STR = "Or, take a photo instead!"
CHOICE_WARN_STR = "Choose between the camera or the file input. The option you\
        choose first will be the input that the model predicts on."
PROJECT_LINK = "Find the model and project repo [here](https://github.com/FFFiend/melanoma_classification)"
RAINBOW="rainbow"
SUCCESS_MSG = "Congratulations! No melanoma detected."
TITLE = "Melanoma Classifier"
TOAST_MSG = "Hey, scroll up!"
UPLOAD_CHOICE_STR = "Uploaded image!"
UPLOAD_STR="Upload your image here."
USAGE_WARNING_MSG = "Please note that while this classifier tries to be as accurate\
    as possible, it is still no substitute to medical advice offered by a certified specialist,\
        dermatologist or surgeon."
WARNING_MSG = "Uh oh, looks like this tested positive for melanoma."

MODEL = None # model.pkl TODO
# MODEL = pickle.load(open('melanoma_CNN.pkl','rb'))

def _predict(*args):
    prediction = None
    if prediction == 0 or prediction is None:
        lit.success(SUCCESS_MSG)
        lit.toast(TOAST_MSG)
    else:
        lit.warning(WARNING_MSG)


lit.title(TITLE,anchor="https://github.com/FFFiend/melanoma_classification")
lit.warning(USAGE_WARNING_MSG)

uploaded_img = lit.file_uploader(UPLOAD_STR)
cam_img = lit.camera_input(CAMERA_UPLOAD_STR)

if cam_img is not None:
    cam_img = np.asarray(Image.open(BytesIO(cam_img.getvalue())))

if uploaded_img is not None:
    uploaded_img = np.asarray(Image.open(BytesIO(uploaded_img.getvalue())))

if cam_img is not None and uploaded_img is not None:
    lit.subheader(CHOICE_WARN_STR, divider=RAINBOW)
    lit.button(UPLOAD_CHOICE_STR, on_click=_predict, args=(list(uploaded_img)))
    lit.button(CAMERA_CHOICE_STR,on_click=_predict, args=(list(cam_img)))

elif cam_img is not None or uploaded_img is not None:
    final_img = None
    if uploaded_img is not None:
        final_img = uploaded_img
        
    if cam_img is not None:
        final_img = cam_img

    lit.button(BUTTON_LABEL,on_click=_predict,args=(list(final_img)))

lit.markdown(PROJECT_LINK,unsafe_allow_html=True)