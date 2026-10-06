import json
import os
from collections import defaultdict


with open("data/train/annotations.json", "r", encoding="utf-8") as f:
    all_data = json.load(f)

with open("splits/val_image_ids.json", "r", encoding="utf-8") as f:
    val_image_ids = json.load(f)


all_data_imgs = all_data["images"]
val_image_ids = set(val_image_ids)

val_images = [img for img in all_data_imgs if img["id"] in val_image_ids and img["source_dataset"] == "DPR_MIR_3963_QUALITY_EVALUATION_09072025"]

images_for_training = [img for img in all_data_imgs if img["id"] not in val_image_ids and img["source_dataset"] == "DPR_MIR_3963_QUALITY_EVALUATION_09072025"]


training_img_file_name = [img["file_name"] for img in images_for_training]

training_img_paths = [os.path.join("data/train/images", img) for img in training_img_file_name]
with open("splits/train_images.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(training_img_paths))


validation_img_file_name = [img["file_name"] for img in val_images]

val_img_paths = [os.path.join("data/train/images", img) for img in validation_img_file_name]
with open("splits/val_images.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(val_img_paths))



LABELS_DIR = "data/train/labels"


def group_boxes_by_image(annotations):
    """Return {image_id: [annotation, ...]} so we can look up a photo's boxes at once."""
    boxes_by_image = defaultdict(list)
    for ann in annotations:
        boxes_by_image[ann["image_id"]].append(ann)
    return boxes_by_image


def coco_box_to_yolo_line(ann, img_w, img_h):
    """Convert one COCO box to one YOLO label line, or None if the box is unusable."""
    x, y, w, h = ann["bbox"]

    # clip the box to the image; YOLO needs all values inside 0..1
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)
    if x2 <= x1 or y2 <= y1:
        return None

    # corners in pixels -> center and size as fractions of THIS image's size
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h

    # COCO classes are 1..369, YOLO wants 0..368
    cls = ann["category_id"] - 1
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def write_yolo_labels(images, boxes_by_image):
    """Write one .txt per image into LABELS_DIR (empty file if the image has no boxes)."""
    os.makedirs(LABELS_DIR, exist_ok=True)
    for img in images:
        lines = []
        for ann in boxes_by_image.get(img["id"], []):
            line = coco_box_to_yolo_line(ann, img["width"], img["height"])
            if line is not None:
                lines.append(line)
        stem = os.path.splitext(img["file_name"])[0]
        with open(os.path.join(LABELS_DIR, stem + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


boxes_by_image = group_boxes_by_image(all_data["annotations"])
write_yolo_labels(images_for_training + val_images, boxes_by_image)