import json
import os
from ultralytics import YOLO

WEIGHTS = "runs/detect/a-real-640/weights/best.pt"   # take the path from the training log
IMGSZ = 640                                     # same as in training

with open("splits/val_ground_truth.json", "r", encoding="utf-8") as f:
    val_images = json.load(f)["images"]

model = YOLO(WEIGHTS)
predictions = []

for img in val_images:
    path = os.path.join("data/train/images", img["file_name"])
    # low conf threshold: mAP needs the whole ranking, not only confident boxes
    result = model.predict(path, imgsz=IMGSZ, conf=0.001, device="mps", verbose=False)[0]

    for (x1, y1, x2, y2), score, cls in zip(
        result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()
    ):
        predictions.append({
            "image_id": img["id"],
            "category_id": int(cls) + 1,             # back from 0..368 to 1..369
            "bbox": [x1, y1, x2 - x1, y2 - y1],      # corners -> [x, y, w, h] in pixels
            "score": score,
        })

with open("splits/val_predictions.json", "w", encoding="utf-8") as f:
    json.dump(predictions, f)

print(f"{len(predictions)} boxes for {len(val_images)} images")