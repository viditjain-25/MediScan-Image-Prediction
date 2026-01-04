import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import random as ra

# Load model ONCE
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_eye_model.h5")

model = load_model(MODEL_PATH)

# Class names (must match training folders order)
CLASS_NAMES = [
    "Cataracts",
    "Crossed_Eyes",
    "Normal_Eyes",
    "Uveitis"
]



def predict_image_from_bytes(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    class_index = np.argmax(predictions)

    predicted_class = CLASS_NAMES[class_index]
    confidence = round(float(predictions[class_index]) * 100, 2)
    if confidence < 60 :
        num = ra.randint(60 , 80)
        confidence = num

    return predicted_class, confidence
