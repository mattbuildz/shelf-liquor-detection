from ultralytics import YOLO

EXPERIMENT_NAME = "a2-adamw-lr001"   # run folder name, change per experiment
EPOCHS = 50
FRACTION = 1.0             # share of the TRAIN list used; 1.0 for a full run
IMGSZ = 640

model = YOLO("yolov8n.pt")
model.train(
    data="scripts/training/dataset.yaml",
    optimizer="AdamW", #changed to AdamW from auto
    lr0=0.001,
    cache="ram",
    epochs=EPOCHS,
    fraction=FRACTION,
    imgsz=IMGSZ,
    batch=16,
    device="mps",           # Apple GPU; use "cpu" if it crashes
    name=EXPERIMENT_NAME,
)