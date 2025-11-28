import streamlit as st
from ultralytics import YOLO
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import cv2
import tempfile
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Load YOLO model
@st.cache_resource
def load_yolo():
    return YOLO("best.pt")  # YOLOv8 model

# Load Keras classifier
@st.cache_resource
def load_classifier():
    return load_model("MobileNetV2-best_model.keras")

yolo_model = load_yolo()
clf_model = load_classifier()

CLASS_NAMES = ["Bird", "Drone"]

st.title("🕊️ Drone & Bird Detection + Classification")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ---------------------------
    # 1️⃣ Classification
    # ---------------------------
    st.subheader("🔍 Classification (Transfer Learning Model)")
    img = image.resize((224, 224))
    arr = img_to_array(img)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, 0)

    preds = clf_model.predict(arr)[0]
    cls = np.argmax(preds)
    conf = preds[cls]

    st.write(f"**Prediction:** {CLASS_NAMES[cls]}")
    st.write(f"**Confidence:** `{conf:.2f}`")

    # ---------------------------
    # 2️⃣ YOLOv8 Detection
    # ---------------------------
    st.subheader("📦 Object Detection (YOLOv8)")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
        image.save(temp.name)
        temp_path = temp.name

    results = yolo_model.predict(temp_path, save=False)
    result = results[0]

    if len(result.boxes) == 0:
        st.warning("No objects detected.")
    else:
        plotted = result.plot()
        plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
        st.image(plotted, use_container_width=True)
