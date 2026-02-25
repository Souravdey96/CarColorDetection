# CarColorDetection
# 🚗 Car Color Detection & Traffic Analytics System

An end-to-end Computer Vision system that detects vehicles in traffic, predicts their colors, draws rule-based bounding boxes, and counts both vehicles and people.

---

## 📌 Project Overview

This project combines **Object Detection + Image Classification + GUI Deployment** into a single intelligent system.

Pipeline:

YOLOv8 → Vehicle Detection  
⬇  
EfficientNetB0 → Car Color Classification  
⬇  
Streamlit → Interactive GUI Visualization

---

## ✨ Features

- Detect cars, buses, and trucks using YOLOv8
- Predict vehicle color using EfficientNetB0 transfer learning
- Rule-based bounding boxes:
  - 🔴 Red box for blue cars
  - 🔵 Blue box for other colors
- Count number of vehicles
- Count number of people at traffic signal
- Streamlit GUI with image preview

---

## 🧠 Tech Stack

- Python
- TensorFlow / Keras
- EfficientNetB0
- YOLOv8 (Ultralytics)
- OpenCV
- Streamlit
- NumPy

---

## 📊 Model Performance

- Training Accuracy: ~92%
- Validation Accuracy: ~85%
- Test Accuracy: ~80%

---

## 🖥️ GUI Preview

(Add screenshots here)

---

## ⚙️ Installation

```bash
git clone https://github.com/Souravdey96/CarColorDetection.git
cd CarColorDetection

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
