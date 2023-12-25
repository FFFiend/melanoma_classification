"""
Frontend logic for streamlit.
"""
import torch
import numpy as np
import streamlit as lit
from io import BytesIO
from model import MelanomaCNN
from PIL import Image
BEST_RESULT_MSG = "TIP: For best results, bring the suspect mole as close to the camera as you can."
BUTTON_LABEL="Go ahead and predict the result using the model!"
CAMERA_CHOICE_STR = "Camera input, please."
CAMERA_UPLOAD_STR = "Or, take a photo instead!"
CHOICE_WARN_STR = "Choose between the camera or the file input. The option you\
        choose first will be the input that the model predicts on."
PROJECT_LINK = "Find the model and project repo [here](https://github.com/FFFiend/melanoma_classification)."
RAINBOW="rainbow"
SUCCESS_MSG = "Congratulations! No melanoma detected."
TITLE = "Melanoma Classifier 🧪 🧬"
TOAST_MSG = "Hey, scroll up!"
UPLOAD_CHOICE_STR = "Uploaded image!"
UPLOAD_STR="Upload your image here."
USAGE_WARNING_MSG = "Please note that while this classifier tries to be as accurate\
    as possible, it is still no substitute to medical advice offered by a certified specialist,\
        dermatologist or surgeon.\
            \n\nAdditionally, this model was trained on a set of 300x300 images, and so inputs\
                are cropped to the aforementioned resolution for now, which might make classification\
                    on camera inputs a bit iffy.\
            \n\nAlso note that the model currently suffers from the Open Set classification problem\
            meaning that until further training/transfer learning, the model has a high misclassification\
                rate on images that aren't of either benign or malignant melanomas. Actively working on\
                    adding this enhancement."
WARNING_MSG = "Uh oh, looks like this tested positive for melanoma."

MODEL = MelanomaCNN()
MODEL.load_state_dict(torch.load("model/model.pt",map_location=torch.device("cpu")))
MODEL.eval()

def _crop_center(img, crop_size):
    h, w, _ = img.shape
    start_h = (h - crop_size[0]) // 2
    start_w = (w - crop_size[1]) // 2
    cropped_img = img[start_h:start_h + crop_size[0], start_w:start_w + crop_size[1], :]
    return cropped_img
    
def _pad(img, target_size):
    h, w, c = img.shape
    pad_h = max(0, target_size[0] - h)
    pad_w = max(0, target_size[1] - w)

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded_img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
    return padded_img

def _predict(*args):
    prediction=0
    img = []
    for row in args:
        img.append(row)
    img = np.array(img)
    if img.shape[2] > 3: 
        img = img[:, :, :3]
    if (img.shape[0]*img.shape[1] > 300*300):
        img = _crop_center(img, (300,300,3))
    elif (img.shape[0]*img.shape[1] < 300*300):
        img = _pad(img, (300,300,3))
    img = np.transpose(img, (2, 0, 1))
    prediction = MODEL(torch.from_numpy(img.astype(np.float32)).unsqueeze(0))
    prediction = prediction.max(1, keepdim=True)[1]
    if prediction==0:
        lit.success(SUCCESS_MSG)
        lit.toast(TOAST_MSG)
    else:
        lit.warning(WARNING_MSG)

lit.title(TITLE)
lit.warning(USAGE_WARNING_MSG)

uploaded_img = lit.file_uploader(UPLOAD_STR)
cam_img = lit.camera_input(CAMERA_UPLOAD_STR)
lit.success(BEST_RESULT_MSG)


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
