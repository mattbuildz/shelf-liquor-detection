#This file is a workaround for checking the results of a model, I needed to take around 10-20% of real images for the testing.
#So the model is trained not on 1123 real images but on about 923.


import json
import random
import os

with open("data/train/annotations.json", "r", encoding="utf-8") as f:
    data = json.load(f)


images = data["images"]

real_images = [img["id"] for img in images if img["source_dataset"] == "DPR_MIR_3963_QUALITY_EVALUATION_09072025"]


print(len(real_images))

random.seed(2)
validation_images = random.sample(real_images, 300)

os.makedirs("splits", exist_ok=True)

with open("splits/val_image_ids.json", "w", encoding="utf-8") as f:
    json.dump(validation_images, f, ensure_ascii=False, indent=2)



