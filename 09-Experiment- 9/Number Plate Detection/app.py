import streamlit as st
import cv2
import torch
import numpy as np
import easyocr
import os
from PIL import Image

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.4  # confidence threshold

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Create folders if they don't exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)

st.title("🚘 Number Plate Detection with OCR")

# File uploader
uploaded_file = st.file_uploader("Upload an image of a car", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Save the uploaded image
    upload_path = os.path.join("uploads", uploaded_file.name)
    with open(upload_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 2. Display the uploaded image
    st.image(upload_path, caption="Uploaded Image", use_container_width=True)

    # 3. Run detection
    img = np.array(Image.open(upload_path).convert("RGB"))
    results = model(img)

    # 4. Render detections onto the image
    annotated = results.render()[0]  # returns a list; take first element
    # YOLOv5 renders in BGR, convert to RGB
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # 5. Save the annotated image to results/
    result_path = os.path.join("results", uploaded_file.name)
    Image.fromarray(annotated_rgb).save(result_path)

    # 6. Display the annotated (detected) image
    st.image(result_path, caption="Detected Plates", use_container_width=True)

    # 7. Perform OCR on each detected bounding box
    st.subheader("📖 Detected Plate Text:")
    for i, det in enumerate(results.xyxy[0]):
        x1, y1, x2, y2 = det[:4].int().tolist()
        crop = img[y1:y2, x1:x2]
        if crop.size:
            ocr_results = reader.readtext(crop)
            for _, text, _ in ocr_results:
                st.success(f"Plate {i+1}: {text}")

    st.info("✅ Done! Upload another image to detect more plates.")
