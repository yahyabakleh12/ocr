from fastapi import FastAPI
from pydantic import BaseModel
import base64
import cv2
import numpy as np
from PIL import Image
from rfdetr import RFDETRBase
import os

# === Configuration ===
CHECKPOINT_PATH = "models/checkpoint_best_total.pth"
# 7 emirate codes followed by digits 0-9 and A-Z for detections
CLASS_NAMES = [
    "AE-AZ", "AE-DU", "AE-SH", "AE-AJ", "AE-RK", "AE-FU", "AE-UQ",
    *[str(i) for i in range(10)],
    *[chr(c) for c in range(ord('A'), ord('Z')+1)]
]
# Emirates where identifier must be numeric only
NUMERIC_IDENTIFIER_EMIRATES = {"AE-AZ", "AE-SH", "AE-AJ"}

# === Point class for storing detection info ===
class Point:
    def __init__(self, x1, y1, x2, y2, label: str, conf: int):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.x = (x1 + x2) // 2
        self.y = (y1 + y2) // 2
        self.label = label
        self.conf = conf

# === Plate parsing logic ===
def parse_plate(points):
    # Define emirate codes
    emirates = {"AE-AZ", "AE-DU", "AE-SH", "AE-AJ", "AE-RK", "AE-FU", "AE-UQ"}
    # Identify city code
    city_pt = next((p for p in points if p.label in emirates), None)
    city = city_pt.label if city_pt else ""
    # Exclude city code from others
    others = [p for p in points if p is not city_pt]
    # If no other detections, return city only
    if not others:
        return city, "", ""
    # Split others into digits and letters
    digits = [p for p in others if p.label.isdigit()]
    letters = [p for p in others if p.label.isalpha() and p.label not in emirates]
    # Non-numeric-identifier emirates: letters allowed
    if city not in NUMERIC_IDENTIFIER_EMIRATES:
        if letters:
            # pick highest-confidence letter
            id_pt = max(letters, key=lambda p: p.conf)
            identifier = id_pt.label
        else:
            # no letter identifier
            identifier = ''
        # number is all digits sorted by x
        number = ''.join(p.label for p in sorted(digits, key=lambda p: p.x))
        return city, identifier, number
    # Numeric-identifier emirates: identifier must be digit or none
    # Prepare sorted digits by x
    digits_sorted = sorted(digits, key=lambda p: p.x)
    # If no digits, nothing to parse
    if not digits_sorted:
        return city, "", ""
    # 1) Vertical check: identifier on top
    ys = [p.y for p in digits_sorted]
    median_y = np.median(ys)
    vertical_thresh = 15
    top = [p for p in digits_sorted if median_y - p.y > vertical_thresh]
    if top:
        # top digits form identifier
        id_grp = sorted(top, key=lambda p: p.x)
        identifier = ''.join(p.label for p in id_grp)
        # remaining digits are number
        rem = [p for p in digits_sorted if p not in top]
        number = ''.join(p.label for p in sorted(rem, key=lambda p: p.x))
        return city, identifier, number
    # 2) Horizontal gap check: identifier on left separated by gap
    xs = [p.x for p in digits_sorted]
    diffs = [xs[i+1] - xs[i] for i in range(len(xs)-1)]
    mean_diff = np.mean(diffs) if diffs else 0
    split_idx = next((i for i, d in enumerate(diffs) if d > mean_diff * 1.5), -1)
    if split_idx >= 0:
        left = digits_sorted[:split_idx+1]
        right = digits_sorted[split_idx+1:]
        # smaller group is identifier
        if len(left) < len(right):
            id_grp, num_grp = left, right
        else:
            id_grp, num_grp = right, left
        identifier = ''.join(p.label for p in sorted(id_grp, key=lambda p: p.x))
        number = ''.join(p.label for p in sorted(num_grp, key=lambda p: p.x))
        return city, identifier, number
    # 3) No identifier present
    return city, "", ''.join(p.label for p in digits_sorted)

# === FastAPI setup ===
app = FastAPI()
model = RFDETRBase(pretrain_weights=CHECKPOINT_PATH, num_classes=len(CLASS_NAMES))

class PlateRequest(BaseModel):
    image_base64: str

@app.post("/detect_plate")
async def detect_plate(request: PlateRequest):
    # Decode image
    data = base64.b64decode(request.image_base64)
    arr = np.frombuffer(data, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    # Inference
    det = model.predict(img_pil, threshold=0.5)
    points = []
    plate_detail = []
    for cid, conf, bb in zip(det.class_id, det.confidence, det.xyxy):
        label = CLASS_NAMES[cid]
        x1, y1, x2, y2 = map(int, bb)
        pt = Point(x1, y1, x2, y2, label, int(conf*100))
        points.append(pt)
        plate_detail.append({
            "character": label,
            "conf": pt.conf,
            "x1": pt.x1,
            "y1": pt.y1,
            "x2": pt.x2,
            "y2": pt.y2
        })
    # Parse plate
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
