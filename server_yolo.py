from fastapi import FastAPI
from pydantic import BaseModel
import base64
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

# === Configuration ===
CHECKPOINT_PATH = "models/best.pt"  
CLASS_NAMES = [
    "AE-AZ", "AE-DU", "AE-SH", "AE-AJ", "AE-RK", "AE-FU", "AE-UQ",
    *[str(i) for i in range(10)],
    *[chr(c) for c in range(ord('A'), ord('Z')+1)]
]
NUMERIC_IDENTIFIER_EMIRATES = {"AE-AZ", "AE-SH", "AE-AJ"}

class Point:
    def __init__(self, x1, y1, x2, y2, label: str, conf: int):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.x = (x1 + x2) // 2
        self.y = (y1 + y2) // 2
        self.label = label
        self.conf = conf

def parse_plate(points):
    emirates = set(CLASS_NAMES[:7])
    city_pt = next((p for p in points if p.label in emirates), None)
    city = city_pt.label if city_pt else ""
    others = [p for p in points if p is not city_pt]
    if not others:
        return city, "", ""
    digits = [p for p in others if p.label.isdigit()]
    letters = [p for p in others if p.label.isalpha() and p.label not in emirates]
    if city not in NUMERIC_IDENTIFIER_EMIRATES:
        identifier = max(letters, key=lambda p: p.conf).label if letters else ""
        number = ''.join(p.label for p in sorted(digits, key=lambda p: p.x))
        return city, identifier, number

    
    digits_sorted = sorted(digits, key=lambda p: p.x)
    if not digits_sorted:
        return city, "", ""

    ys = [p.y for p in digits_sorted]
    median_y = np.median(ys)
    vertical_thresh = 15
    top = [p for p in digits_sorted if median_y - p.y > vertical_thresh]
    if top:
        id_grp = sorted(top, key=lambda p: p.x)
        rem = [p for p in digits_sorted if p not in top]
        return city, ''.join(p.label for p in id_grp), ''.join(p.label for p in sorted(rem, key=lambda p: p.x))

    xs = [p.x for p in digits_sorted]
    diffs = [xs[i+1] - xs[i] for i in range(len(xs)-1)]
    mean_diff = np.mean(diffs) if diffs else 0
    split_idx = next((i for i, d in enumerate(diffs) if d > mean_diff * 1.5), -1)
    if split_idx >= 0:
        left, right = digits_sorted[:split_idx+1], digits_sorted[split_idx+1:]
        id_grp, num_grp = (left, right) if len(left) < len(right) else (right, left)
        return city, ''.join(p.label for p in sorted(id_grp, key=lambda p: p.x)), ''.join(p.label for p in sorted(num_grp, key=lambda p: p.x))

    return city, "", ''.join(p.label for p in digits_sorted)

# === FastAPI setup ===
app = FastAPI()
yolo_model = YOLO(CHECKPOINT_PATH)  # تحميل نموذج YOLO المدرب

class PlateRequest(BaseModel):
    image_base64: str

@app.post("/detect_plate")
async def detect_plate(request: PlateRequest):
    # فك ترميز الصورة من base64
    data = base64.b64decode(request.image_base64)
    arr = np.frombuffer(data, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # تنبؤ YOLO
    results = yolo_model.predict(img_bgr, conf=0.5)[0]  # نأخذ النتيجة الأولى فقط

    points = []
    plate_detail = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0]) * 100
        cls_id = int(box.cls[0])
        label = CLASS_NAMES[cls_id]
        pt = Point(x1, y1, x2, y2, label, int(conf))
        points.append(pt)
        plate_detail.append({
            "character": label,
            "conf": pt.conf,
            "x1": pt.x1,
            "y1": pt.y1,
            "x2": pt.x2,
            "y2": pt.y2
        })

    # تجميع مكونات اللوحة
    city, identifier, number = parse_plate(points)
    full_plate = f"{city} {identifier} {number}".strip()
    total_conf = int(np.mean([p.conf for p in points])) if points else 0

    return {
        "city_name": city,
        "identifier": identifier,
        "number": number,
        "full_plate": full_plate,
        "total_confidence": total_conf,
        "plate_detail": plate_detail
    }
