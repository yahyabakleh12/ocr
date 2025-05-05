import cv2
import numpy as np
from PIL import Image
from rfdetr import RFDETRBase
import supervision as sv

# === CONFIG ===
IMAGE_PATH = "507260.jpg"
CHECKPOINT_PATH = "checkpoint_best_total.pth"

# Custom class names (43 classes)
CLASS_NAMES = [
    "AE-AZ", "AE-DU", "AE-SH", "AE-AJ", "AE-RK", "AE-FU", "AE-UQ",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
]

# === Load model with checkpoint ===
model = RFDETRBase(pretrain_weights=CHECKPOINT_PATH, num_classes=len(CLASS_NAMES))

# === Load and prepare image ===
image_bgr = cv2.imread(IMAGE_PATH)
image_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

# === Run prediction ===
detections = model.predict(image_pil, threshold=0.5)

# === Prepare visualization ===
text_scale = sv.calculate_optimal_text_scale(resolution_wh=image_pil.size)
thickness = sv.calculate_optimal_line_thickness(resolution_wh=image_pil.size)
palette = sv.ColorPalette.from_hex([
    "#ff3838", "#ff9d97", "#ff701f", "#ffb21d", "#cfdd3f", "#38ff45",
    "#39c5bb", "#00cfff", "#0086ff", "#7139ff", "#c17aff", "#ff5eb0"
])

bbox_annotator = sv.BoxAnnotator(color=palette, thickness=thickness)
label_annotator = sv.LabelAnnotator(
    color=palette,
    text_color=sv.Color.BLACK,
    text_scale=text_scale,
    smart_position=True
)

# === Generate labels ===
labels = [
    f"{CLASS_NAMES[class_id]} {confidence:.2f}"
    for class_id, confidence in zip(detections.class_id, detections.confidence)
]

# === Annotate ===
annotated_img = bbox_annotator.annotate(image_pil.copy(), detections)
annotated_img = label_annotator.annotate(annotated_img, detections, labels)

# === Display the result ===
annotated_bgr = cv2.cvtColor(np.array(annotated_img), cv2.COLOR_RGB2BGR)
cv2.imshow("Detection Result", annotated_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
