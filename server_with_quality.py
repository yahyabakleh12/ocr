from fastapi import FastAPI
from pydantic import BaseModel
import base64
import cv2
import numpy as np
from PIL import Image
from rfdetr import RFDETRBase
import time
import torch
import RRDBNet_arch as arch
import os

# Configuration
CHECKPOINT_PATH = "models/checkpoint_best_total.pth"
ESRGAN_MODEL_PATH = "models/RRDB_ESRGAN_x4.pth"
CLASS_NAMES = [
    "AE-AZ", "AE-DU", "AE-SH", "AE-AJ", "AE-RK", "AE-FU", "AE-UQ",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
]

# Load models
model = RFDETRBase(pretrain_weights=CHECKPOINT_PATH, num_classes=len(CLASS_NAMES))
esrgan_model = arch.RRDBNet(3, 3, 64, 23, gc=32)
esrgan_model.load_state_dict(torch.load(ESRGAN_MODEL_PATH), strict=True)
esrgan_model.eval()
esrgan_model = esrgan_model.to("cuda" if torch.cuda.is_available() else "cpu")

# FastAPI app
app = FastAPI()

# Request schema
class PlateRequest(BaseModel):
    image_base64: str

@app.post("/detect_plate")
async def detect_plate(request: PlateRequest):
    start_time = time.time()

    # Decode base64 image
    image_data = base64.b64decode(request.image_base64)
    np_img = np.frombuffer(image_data, np.uint8)
    img_bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # === Save original image for debugging ===
    os.makedirs('debug', exist_ok=True)
    debug_time = int(time.time())
    cv2.imwrite(f'debug/{debug_time}_before.jpg', img_bgr)

    # === Enhance Image Quality ===
    img = img_bgr * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to("cpu")

    with torch.no_grad():
        output = esrgan_model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)

    # Save enhanced image for debugging
    cv2.imwrite(f'debug/{debug_time}_after.jpg', output)

    img_pil = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

    # Predict
    detections = model.predict(img_pil, threshold=0.5)

    # Build plate character list
    plate_detail = []
    city_name = ""
    character_part = ""
    number_part = ""

    for class_id, conf, xyxy in zip(detections.class_id, detections.confidence, detections.xyxy):
        label = CLASS_NAMES[class_id]
        x1, y1, x2, y2 = map(int, xyxy)
        plate_detail.append({
            "character": label,
            "conf": int(conf * 100),
            "x_center": (x1 + x2) // 2
        })

    # Sort left to right based on x_center
    plate_detail.sort(key=lambda c: c["x_center"])

    for c in plate_detail:
        if c["character"].startswith("AE-"):
            city_name = c["character"]
        elif c["character"].isalpha():
            character_part += c["character"]
        elif c["character"].isdigit():
            number_part += c["character"]

    plate_string = f"{city_name} {character_part} {number_part}"

    total_confidence = int(sum([c["conf"] for c in plate_detail]) / len(plate_detail)) if plate_detail else 0

    return {
        "result": plate_string,
        "city_name": city_name,
        "character": character_part,
        "number": number_part,
        "plate_detail": plate_detail,
        "total_confidence": total_confidence
    }
