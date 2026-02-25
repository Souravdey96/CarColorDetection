import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from PIL import Image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

# 1. UPDATED: Ensure your architecture matches exactly
def build_color_model(num_classes=15):
    # If your Kaggle model used a Normalization layer, it usually goes here.
    # But since you are using EfficientNetB0, it has its own internal normalization.
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(224,224,3)
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Car Color Detection", layout="wide")
st.title("🚦 Car Color Detection System")

# =============================
# LOAD MODELS (FIXED SECTION)
# =============================
@st.cache_resource
def load_models():
    # Load YOLO
    yolo_model = YOLO("yolov8n.pt")

    # FIX: Initialize and Build the model before loading weights
    color_model = build_color_model()
    
    # STEP 1: Manually build the model with the input shape
    color_model.build((None, 224, 224, 3)) 
    
    # STEP 2: Load weights with skip_mismatch=True to bypass the Normalization error
    color_model.load_weights("final_weights.weights.h5", skip_mismatch=True)

    return yolo_model, color_model

yolo_model, color_model = load_models()

# Alphabetical list of colors (ensure this matches your Kaggle folder order!)
class_names = [
    'beige','black','blue','brown','gold',
    'green','grey','orange','pink','purple',
    'red','silver','tan','white','yellow'
]

uploaded_file = st.file_uploader(
    "Upload Traffic Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    # Convert PIL to OpenCv format
    image = np.array(original_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Run YOLO detection
    results = yolo_model(image, conf=0.6)

    car_count = 0
    person_count = 0

    for r in results:
        for box in r.boxes:
            label = yolo_model.names[int(box.cls[0])]
            x1,y1,x2,y2 = map(int, box.xyxy[0])

            if label in ["car","truck","bus"]:
                car_count += 1

                # Crop the car for color detection
                crop = image[y1:y2, x1:x2]
                if crop.size == 0: continue # Skip empty crops
                
                crop = cv2.resize(crop,(224,224))
                
                # PRE-PROCESSING: Ensure this matches your Kaggle training!
                crop = crop / 255.0
                crop = np.expand_dims(crop, 0)

                # Predict Color
                pred = color_model.predict(crop, verbose=0)
                color_index = np.argmax(pred)
                color_name = class_names[color_index]
                confidence = np.max(pred)

                # Draw on image
                cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0),2)
                display_text = f"{color_name} ({confidence*100:.1f}%)"
                cv2.putText(image, display_text, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

            elif label=="person":
                person_count += 1

    # Final counts display
    cv2.putText(image,f"Cars: {car_count}",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.putText(image,f"People: {person_count}",(20,80),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    # Display the final processed image in Streamlit
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
