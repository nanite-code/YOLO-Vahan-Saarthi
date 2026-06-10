import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "new.onnx"
model = YOLO(MODEL_PATH, task='detect')

# Create a dummy image
frame = np.zeros((480, 640, 3), dtype=np.uint8)

frame_resized = cv2.resize(frame, (416, 416))
print("Predicting...")
try:
    results = model.predict(source=frame_resized, imgsz=416, verbose=False)
    r = results[0]
    print(f"Boxes: {len(r.boxes)}")
    for box in r.boxes:
        cls_id = int(box.cls[0].item())
        print(f"Class: {cls_id}")
except Exception as e:
    import traceback
    traceback.print_exc()
