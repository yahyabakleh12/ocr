import cv2
import numpy as np
from PIL import Image
from rfdetr import RFDETRBase

# === CONFIG ===
IMAGE_PATH = "data/t6.jpg"
CHECKPOINT_PATH = "models/ch.pth"

# Class mapping (43 classes)
CLASS_NAMES = [
    "AE-AZ", "AE-DU", "AE-SH", "AE-AJ", "AE-RK", "AE-FU", "AE-UQ",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
]

# === Load model with custom checkpoint ===
model = RFDETRBase(pretrain_weights=CHECKPOINT_PATH, num_classes=len(CLASS_NAMES))

# === Load and prepare image ===
image_bgr = cv2.imread(IMAGE_PATH)
image_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

# === Predict ===
detections = model.predict(image_pil, threshold=0.5)

# === Extract character predictions and their X-center positions ===
plate_characters = []
for class_id, confidence, xyxy in zip(detections.class_id, detections.confidence, detections.xyxy):
    x_center = (xyxy[0] + xyxy[2]) / 2  # x1 + x2 / 2
    y_center = (xyxy[1] + xyxy[3]) / 2  # x1 + x2 / 2
    label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else "unknown"
    plate_characters.append((x_center, label))

# === Sort characters left to right ===
plate_characters.sort(key=lambda x: x[0])
plate_string = ''.join([char for _, char in plate_characters])

# === Print result ===
print(f"Detected plate: {plate_string}")
