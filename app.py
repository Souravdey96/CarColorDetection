import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

# =====================================
# BUILD COLOR MODEL
# =====================================
def build_color_model(num_classes=15):

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


# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(page_title="Car Color Detection", layout="wide")

st.title("🚦 Car Color Detection System")
st.write("Detects vehicle colors, counts vehicles, and counts people in traffic scenes.")

# =====================================
# LOAD MODELS (CACHED)
# =====================================
@st.cache_resource
def load_models():

    yolo_model = YOLO("yolov8n.pt")

    color_model = build_color_model()
    color_model.load_weights("color_model.weights.h5")

    return yolo_model, color_model


yolo_model, color_model = load_models()

class_names = [
    'beige','black','blue','brown','gold',
    'green','grey','orange','pink','purple',
    'red','silver','tan','white','yellow'
]

# =====================================
# IMAGE UPLOAD
# =====================================
uploaded_file = st.file_uploader(
    "Upload Traffic Image",
    type=["jpg","jpeg","png"]
)

# =====================================
# MAIN LOGIC
# =====================================
if uploaded_file is not None:

    original_image = Image.open(uploaded_file)
    st.subheader("Original Image")
    st.image(original_image, use_column_width=True)

    image = np.array(original_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    results = yolo_model(image, conf=0.6)

    car_count = 0
    person_count = 0

    for r in results:
        for box in r.boxes:

            confidence = float(box.conf[0])
            if confidence < 0.6:
                continue

            cls_id = int(box.cls[0])
            label = yolo_model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            width = x2 - x1
            height = y2 - y1


            if width < 70 or height < 70:
                continue

            # =============================
            # VEHICLE LOGIC
            # =============================
            if label in ["car","truck","bus"]:

                car_count += 1

                car_crop = image[y1:y2, x1:x2]
                car_crop = cv2.resize(car_crop, (224,224))
                car_crop = car_crop / 255.0
                car_crop = np.expand_dims(car_crop, axis=0)

                pred = color_model.predict(car_crop, verbose=0)

                confidence_score = np.max(pred)
                pred_class = np.argmax(pred)


                if confidence_score < 0.70:
                    color_name = "unknown"
                else:
                    color_name = class_names[pred_class]


                if color_name == "blue":
                    box_color = (0,0,255)   
                else:
                    box_color = (255,0,0)   


                cv2.rectangle(image, (x1,y1), (x2,y2), box_color, 2)


                if color_name != "unknown":
                    cv2.putText(
                        image,
                        color_name,
                        (x1, y1-8 if y1>20 else y1+20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        box_color,
                        2
                    )

            # =============================
            # PERSON LOGIC
            # =============================
            elif label == "person":
                person_count += 1

    # =====================================
    # TOP LEFT COUNTERS
    # =====================================
    cv2.putText(
        image, f"Cars: {car_count}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.putText(
        image, f"People: {person_count}",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.subheader("Processed Output")
    st.image(processed_image, use_column_width=True)

    st.success(f"Total Cars Detected: {car_count}")
    st.success(f"Total People Detected: {person_count}")