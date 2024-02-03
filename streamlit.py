import streamlit as st
from PIL import Image
from rembg import remove 
import tensorflow as tf
import numpy as np

st.title("Schoki needed?")
isHappy = None

st.markdown("<div style='border: solid red 4px; height: 224px; width: 224px;position: absolute;z-index: 999;left: 50%;transform: translateX(-50%);top: 130px;}'></div>", unsafe_allow_html=True)
img_file_buffer = st.camera_input("Take a picture")
# st.markdown('<style>div[data-testid="stCameraInput"] video{object-fit: none;border-top: 105px solid grey;border-bottom: 70px solid grey;border-right: 240px solid grey;border-left: 240px solid grey;}</style>', unsafe_allow_html=True)

loaded_model = tf.keras.models.load_model('models/best_model.h5')
prediction = None
image= None
isHappy = None
if img_file_buffer is not None:
    image_oi = Image.open(img_file_buffer)
    image_oi = remove(image_oi)

    image_oi = image_oi.convert("RGB")

    # image_oi = Image.open("./dataset/Validation/happy/flipped_casual-urban-guy_75922-512.jpg_face_0.jpg")
   
    # Resize the image to the target size (e.g., 224x224)
    target_size = (224, 224)
    width, height = image_oi.size 
    left = (width - target_size[0])/2
    top = (height - target_size[1])/2
    right = (width + target_size[0])/2
    bottom = (height + target_size[1])/2
    image = image_oi.crop((left, top, right, bottom))
    # image = image_oi.resize(target_size)

    # Convert to NumPy array
    image_array = np.array(image)

    # Normalize pixel values to the range [0, 1]
    image_array = image_array / 255.0

    # Expand dimensions to match the input shape expected by your CNN
    image_array = np.expand_dims(image_array, axis=0)

    # Assuming loaded_model is your pre-trained CNN model
    prediction = loaded_model.predict(image_array)

if prediction:
    isHappy = prediction < 0.5

if isHappy != None:
    if isHappy :
        st.header(":green[Schön, dass es dir gut geht!]")
    else :
        st.header(':red[Ich glaube, du brauchst Schokolade. Immer wenn du hungrig bist, wirst du zur Diva! 🍫]')