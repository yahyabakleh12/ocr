from fastapi import FastAPI
from pydantic import BaseModel
import base64
import cv2
import numpy as np
from PIL import Image
from rfdetr import RFDETRBase
import time
import os

# === Config ===
CHECKPOINT_PATH = "models/checkpoint_best_total.pth"
CLASS_NAMES = [
    "AE-AZ", "AE-DU", "AE-SH", "AE-AJ", "AE-RK", "AE-FU", "AE-UQ",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
]

# === Load model ===
model = RFDETRBase(pretrain_weights=CHECKPOINT_PATH, num_classes=len(CLASS_NAMES))

# === FastAPI setup ===
app = FastAPI()

class PlateRequest(BaseModel):
    image_base64: str

@app.post("/detect_plate")
async def detect_plate(request: PlateRequest):
    image_data = base64.b64decode(request.image_base64)
    np_img = np.frombuffer(image_data, np.uint8)
    img_bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    detections = model.predict(img_pil, threshold=0.5)

    plate_detail = []
    for class_id, conf, xyxy in zip(detections.class_id, detections.confidence, detections.xyxy):
        label = CLASS_NAMES[class_id]
        x1, y1, x2, y2 = map(int, xyxy)
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2
        plate_detail.append({
            "character": label,
            "conf": int(conf * 100),
            "x_center": x_center,
            "y_center": y_center
        })
    print(plate_detail)
    parsed = parse_plate_by_logic(plate_detail)
    total_conf = int(sum([p["conf"] for p in plate_detail]) / len(plate_detail)) if plate_detail else 0

    return {
        "result": parsed.get("full_plate", ""),
        "city_name": parsed.get("city_name", ""),
        "identifier": parsed.get("identifier", ""),
        "number": parsed.get("number", ""),
        "total_confidence": total_conf,
        "plate_detail": plate_detail
    }

# === Logic engine based on emirate spacing ===
def parse_plate_by_logic(plate_detail):
    emirate_codes = {"AE-AZ", "AE-DU", "AE-SH", "AE-AJ", "AE-RK", "AE-FU", "AE-UQ"}

    if not plate_detail:
        return {"error": "No detections"}

    plate_detail.sort(key=lambda x: x["x_center"])

    city = next((c for c in plate_detail if c["character"] in emirate_codes), None)
    if not city:
        return {"error": "City code not found"}

    city_name = city["character"]
    remaining = [c for c in plate_detail if c != city]

    if not remaining:
        return {
            "city_name": city_name,
            "identifier": "",
            "number": "",
            "full_plate": f"{city_name}"
        }

    y_vals = [c["y_center"] for c in remaining]
    main_y = int(np.median(y_vals)) if y_vals else 0
    threshold = 15
    main_line = [c for c in remaining if abs(c["y_center"] - main_y) <= threshold]
    main_line.sort(key=lambda c: c["x_center"])

    x_centers = [c["x_center"] for c in main_line]
    avg_spacing = np.diff(x_centers).mean() if len(x_centers) > 1 else 0

    identifier = ""
    number_part = ""

    contains_alpha = any(c["character"].isalpha() for c in main_line)
    contains_digit = any(c["character"].isdigit() for c in main_line)

    if contains_alpha and contains_digit:
        alpha_group = [c for c in main_line if c["character"].isalpha()]
        digit_group = [c for c in main_line if c["character"].isdigit()]
        if alpha_group[0]["x_center"] < digit_group[0]["x_center"]:
            identifier = ''.join([c["character"] for c in alpha_group])
            number_part = ''.join([c["character"] for c in digit_group])
        else:
            number_part = ''.join([c["character"] for c in digit_group])
            identifier = ''.join([c["character"] for c in alpha_group])
    elif contains_digit:
        split_index = 0
        for i in range(1, len(main_line)):
            gap = main_line[i]["x_center"] - main_line[i-1]["x_center"]
            if gap > avg_spacing * 1.6:
                split_index = i
                break
        identifier_group = main_line[:split_index]
        number_group = main_line[split_index:]
        identifier = ''.join([c["character"] for c in identifier_group])
        number_part = ''.join([c["character"] for c in number_group])
    else:
        identifier = ''.join([c["character"] for c in main_line])

    return {
        "city_name": city_name,
        "identifier": identifier,
        "number": number_part,
        "full_plate": f"{city_name} {identifier} {number_part}".strip()
    }
