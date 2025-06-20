import base64
import json
import os
import time
from datetime import datetime

import cv2
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import YOLO


# === Helper Functions ===
def get_centroid(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) // 2, (y1 + y2) // 2


def compute_iou(box_a, box_b):
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    inter = (x_right - x_left) * (y_bottom - y_top)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def filter_overlapping_detections(detections, iou_thresh=0.5):
    filtered = []
    for det in detections:
        keep = True
        for f in filtered:
            if compute_iou(det["bbox"], f["bbox"]) > iou_thresh:
                if det["conf"] > f["conf"]:
                    f.update(det)
                keep = False
                break
        if keep:
            filtered.append(det)
    return filtered


def detect_and_crop_plate(image_bgr, detector):
    results = detector.predict(image_bgr)[0]
    plates = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        plates.append({"bbox": [x1, y1, x2, y2], "conf": conf})
    plates = filter_overlapping_detections(plates)
    crops = []
    for p in plates:
        x1, y1, x2, y2 = p["bbox"]
        crop = image_bgr[y1:y2, x1:x2]
        crops.append((crop, p["bbox"], p["conf"]))
    return crops


# === Models ===
LICENSE_PLATE_MODEL = os.path.join("models", "license_plate_detector.pt")
RECOGNITION_MODEL = os.path.join("models", "recognition_model.pt")
license_plate_detector = YOLO(LICENSE_PLATE_MODEL)
recognition_model = YOLO(RECOGNITION_MODEL)


# === FastAPI setup ===
app = FastAPI()


class PlateRequest(BaseModel):
    image_base64: str


@app.post("/detect_plate")
async def detect_plate(request: PlateRequest):
    data = base64.b64decode(request.image_base64)
    arr = np.frombuffer(data, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    plates = detect_and_crop_plate(img_bgr, license_plate_detector)
    if not plates:
        return {
            "plate_number": "",
            "plate_code": "",
            "plate_city": "",
            "confidence": 0,
            "response": []
        }

    # Use the first detected plate
    plate_img, bbox, plate_conf = plates[0]
    rec_results = recognition_model.predict(plate_img)[0]

    characters = []
    for r in rec_results.boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        cls_id = int(r.cls[0])
        label = recognition_model.names.get(cls_id, str(cls_id)) if hasattr(recognition_model, "names") else str(cls_id)
        conf = float(r.conf[0])
        characters.append({
            "character": label,
            "conf": int(conf * 100),
            "x1": x1 + bbox[0],
            "y1": y1 + bbox[1],
            "x2": x2 + bbox[0],
            "y2": y2 + bbox[1]
        })

    characters.sort(key=lambda c: get_centroid([c["x1"], c["y1"], c["x2"], c["y2"]])[0])

    plate_city = ""
    plate_code = ""
    plate_number = ""
    for c in characters:
        lbl = c["character"]
        if lbl.startswith("AE-") and not plate_city:
            plate_city = lbl
        elif lbl.isalpha() and not lbl.startswith("AE-"):
            plate_code += lbl
        elif lbl.isdigit():
            plate_number += lbl

    avg_conf = int(sum(ch["conf"] for ch in characters) / len(characters)) if characters else int(plate_conf * 100)

    return {
        "plate_number": plate_number,
        "plate_code": plate_code,
        "plate_city": plate_city,
        "confidence": avg_conf,
        "response": {
            "box": {
                "x1": bbox[0],
                "y1": bbox[1],
                "x2": bbox[2],
                "y2": bbox[3]
            },
            "plate_detail_list": characters
        }
    }
