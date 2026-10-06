import json
import os
import yaml

with open("data/train/annotations.json", "r", encoding="utf-8") as f:
    categories = json.load(f)["categories"]

# COCO ids are 1..369, YOLO class indices are 0..368
names = {c["id"] - 1: c["name"] for c in categories}

cfg = {
    "train": os.path.abspath("splits/train_images.txt"),
    "val": os.path.abspath("splits/val_images.txt"),
    "names": names,
}

with open("scripts/training/dataset.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)