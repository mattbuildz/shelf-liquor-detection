import json

with open("splits/val_image_ids.json", "r", encoding="utf-8") as f:
    val_image_ids = json.load(f)


with open("data/train/annotations.json", "r", encoding="utf-8") as f:
    data = json.load(f)

images = data["images"]

real_images = [img["id"] for img in images if img["source_dataset"] == "DPR_MIR_3963_QUALITY_EVALUATION_09072025"]

print(len(real_images))

val_image_ids = set(val_image_ids)
real_images = set(real_images)

train_images = real_images - val_image_ids
print(len(train_images))

#how many classes from 369 is located on real photos
train_categories = {cat["category_id"] for cat in data["annotations"] if cat["image_id"] in train_images}
val_categories = {cat["category_id"] for cat in data["annotations"] if cat["image_id"] in val_image_ids}
real_categories = {cat["category_id"] for cat in data["annotations"] if cat["image_id"] in real_images}

print(f"TRAIN CLASSES: {len(train_categories)}")
print(f"VALIDATION CLASSES: {len(val_categories)}")
print(f"REAL CLASSES: {len(real_categories)}")


print(f"ROZNICA: {len(train_categories - val_categories)}")
