# ocr
ocr server

# models
 # rfdetr
 you can download the model frome google drive here ["https://drive.google.com/file/d/1W00nAp7R4gtsVrBhtga0qBAuN5eI68CF/view?usp=sharing"]
 # Yolo V11
 you can download the model frome google drive here ["https://drive.google.com/file/d/1Ju7dJ2-ekxmF_WPdTsis0dNjYBKIZtZR/view?usp=sharing"]
## Running main_server

Place the YOLO detection and recognition models inside the `models/` directory with the names `license_plate_detector.pt` and `recognition_model.pt`. Install the dependencies from `requirements.txt` and run:

```bash
uvicorn main_server:app --reload
```

The service exposes a `/detect_plate` endpoint that accepts a base64 encoded image.
