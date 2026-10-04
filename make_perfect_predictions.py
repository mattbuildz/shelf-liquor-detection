import json


with open("splits/val_ground_truth.json", "r", encoding="utf-8") as f:
    answers = json.load(f)



frames = answers["annotations"]

ideal_list = [{
    "image_id":frame["image_id"],
    "category_id":frame["category_id"],
    "bbox":frame["bbox"],
    "score":1.0
} for frame in frames]

print(len(ideal_list))




with open("splits/val_perfect_predictions.json", "w", encoding="utf-8") as f:
    json.dump(ideal_list, f, ensure_ascii=False, indent=2)




