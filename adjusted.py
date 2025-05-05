import cv2
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
import supervision as sv
import numpy as np
from PIL import Image
from bytetracker import ByteTracker

# Initialize the RFDETR model
model = RFDETRBase()

# Open the video file
cap = cv2.VideoCapture("cars.mp4")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

# Define color palette
color = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
])

# Initialize ByteTracker
tracker = ByteTracker(frame_rate=fps)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert to RGB (PIL format)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Run object detection
    detections = model.predict(image, threshold=0.5)

    # Filter detections for cars (class ID 3 is typically "car" in COCO)
    car_detections = [
        (detection, idx) for idx, detection in enumerate(detections)
        if detections.class_id[idx] == 3  # class_id 3 corresponds to "car"
    ]

    # Prepare the detection boxes and confidence scores
    dets = []
    for detection, idx in car_detections:
        # Access the bounding boxes from detections.xyxy
        bbox = detections.xyxy[idx]  # Access the bounding box directly from detections
        x1, y1, x2, y2 = bbox  # Extract coordinates
        confidence = detections.confidence[idx]
        dets.append([x1, y1, x2, y2, confidence])

    # Use ByteTracker to track detections
    online_targets = tracker.update(np.array(dets))

    # Annotate the image
    bbox_annotator = sv.BoxAnnotator(color=color, thickness=2)
    label_annotator = sv.LabelAnnotator(
        color=color,
        text_color=sv.Color.BLACK,
        text_scale=0.5,
        smart_position=True
    )

    # Draw updated car bounding boxes
    for target in online_targets:
        x1, y1, x2, y2, _ = target  # Extract the coordinates
        label = f"{COCO_CLASSES[3]}"  # '3' is the ID for "car" in COCO

        # Annotate the bounding box and label
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Write the frame to the output video
    out.write(frame)

    # Display the result
    cv2.imshow('Car Tracking', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()  # Release the VideoWriter
cv2.destroyAllWindows()
