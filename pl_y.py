import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# === CONFIG ===
IMAGE_PATH = "data/1.jpg"
CHECKPOINT_PATH = "models/best.pt"  # <-- change to your trained model

CLASS_NAMES = [
    "AE-AZ", "AE-DU", "AE-SH", "AE-AJ", "AE-RK", "AE-FU", "AE-UQ",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
]

# === Load YOLOv11 Model ===
model = YOLO(CHECKPOINT_PATH)

# === Load Image ===
image_bgr = cv2.imread(IMAGE_PATH)

# === Predict ===
results = model.predict(image_bgr, conf=0.5)[0]

# === Collect Detections ===
plate_characters = []

for box in results.boxes:
    class_id = int(box.cls[0].item())
    print(class_id)
    label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else "unknown"
    conf = float(box.conf[0].item()) * 100
    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
    x_center = (x1 + x2) / 2
    plate_characters.append({
        "label": label,
        "conf": conf,
        "x_center": x_center
    })

# === Sort left to right ===
plate_characters.sort(key=lambda c: c["x_center"])

# === Plate Construction ===
city_name = ""
character_part = ""
number_part = ""

for c in plate_characters:
    if c["label"].startswith("AE-") and city_name == "":  # pick only the first emirate
        city_name = c["label"]
    elif c["label"].isalpha() and not c["label"].startswith("AE-"):
        character_part += c["label"]
    elif c["label"].isdigit():
        number_part += c["label"]

# === Compose plate string ===
plate_string = f"{city_name} {character_part} {number_part}"

# === Print result ===
# print(f"✅ Detected plate: {plate_string}")

