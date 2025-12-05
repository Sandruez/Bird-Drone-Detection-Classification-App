# 🚁 Aerial Object Classification & Detection: Bird vs. Drone

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bird-drone-detection.streamlit.app/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue)](https://dagshub.com/chandrapapr1501/aerial-detector-v1.mlflow/#/experiments/0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20YOLOv8-orange)](https://www.tensorflow.org/)

## 📌 Project Overview
This project builds an intelligent Deep Learning system to distinguish between **Birds** and **Drones** in aerial imagery. Distinguishing between biological and mechanical aerial objects is critical for **security surveillance**, **wildlife protection**, and **airspace safety**.

The solution performs two main tasks:
1.  **Image Classification:** Classifies input images as either "Bird" or "Drone" using Custom CNN and Transfer Learning models (MobileNetV2, ResNet50, EfficientNetB0).
2.  **Object Detection:** Localizes and labels birds or drones in images using **YOLOv8**.

The final system is deployed as an interactive **Streamlit** web application.

---

## 🎯 Key Objectives
* **Classify** aerial images as Bird or Drone with high accuracy.
* **Detect** and localize objects in real-time scenes.
* **Deploy** a user-friendly web interface for easy accessibility.
* **Track** experiments and model performance using MLflow.

---

## 🚀 Real-Time Business Use Cases
* **🦅 Wildlife Protection:** Monitor bird activity near wind farms/airports to prevent collisions.
* **🔐 Security & Defense:** Identify unauthorized drones entering restricted airspace.
* **✈️ Airport Safety:** Prevent bird strikes by tracking movements around runways.
* **🌍 Environmental Research:** Automate bird population tracking from aerial footage.

---

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow/Keras, PyTorch, Ultralytics YOLOv8
* **Computer Vision:** OpenCV, Pillow
* **Web Framework:** Streamlit
* **Experiment Tracking:** MLflow (DagsHub)
* **Language:** Python

---

## 📂 Dataset Details

### 1. Classification Dataset
* **Task:** Binary Classification (Bird / Drone)
* **Format:** RGB Images (`.jpg`)
* **Data Split:**
    * **Train:** 2,662 images (Bird: 1414, Drone: 1248)
    * **Validation:** 442 images
    * **Test:** 215 images

### 2. Object Detection Dataset
* **Format:** YOLOv8 (images with `.txt` annotations)
* **Total Images:** 3,319
* **Annotations:** Bounding boxes `<class_id> <x_center> <y_center> <width> <height>`

---

## ⚙️ Methodology

### 1. Data Preprocessing
* **Resizing:** Images resized to `224x224` (CNN) and `640x640` (YOLO).
* **Normalization:** Pixel values scaled to `[0, 1]`.
* **Augmentation:** RandomFlip, RandomRotation, RandomZoom, and RandomContrast applied to handle variations.

### 2. Model Architecture
* **Custom CNN:** Built from scratch with Conv2D, MaxPooling, and Dropout layers.
* **Transfer Learning:** Fine-tuned **MobileNetV2**, **ResNet50**, and **EfficientNetB0**.
* **YOLOv8s:** Trained for 50 epochs for robust object detection.

---

## 📊 Performance Results (YOLOv8)
The detection model achieved the following on the validation set:
* **Precision:** 0.886
* **Recall:** 0.737
* **mAP50:** 0.810
* **mAP50-95:** 0.529

---

## 💻 Installation & Usage

### Prerequisites
Ensure you have Python installed. Then, clone the repository:

```bash
git clone [https://github.com/Sandruez/Bird-Drone-Detection-Classification-App.git](https://github.com/Sandruez/Bird-Drone-Detection-Classification-App.git)
cd Bird-Drone-Detection-Classification-App
