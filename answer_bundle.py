import json

with open("splits/val_image_ids.json", "r", encoding="utf-8") as f:
    val_ids = json.load(f)

with open("data/train/annotations.json", "r", encoding="utf-8") as f:
    all_data = json.load(f)


id_searched = all_data["images"]
id_wanted = set(val_ids)
id_outcome = [img for img in id_searched if img["id"] in id_wanted]


print(f"SEARCHED: {len(id_searched)}")
print(f"WANTED: {len(id_wanted)}")
print(f"OUTCOME: {len(id_outcome)}")

print(f"CATEGORIES: {len(all_data["categories"])}")


frame_searched = all_data["annotations"]

frame_outcome = [frame for frame in frame_searched if frame["image_id"] in id_wanted]

print(f"FRAMES: {len(frame_searched)}")
print(f"FRAMES OUTCOME: {len(frame_outcome)}")


dictionary = {
    "images":id_outcome,
    "annotations":frame_outcome,
    "categories":all_data["categories"]
    }



with open("splits/answer_bundle.json", "w", encoding="utf-8") as f:
    json.dump(dictionary, f, ensure_ascii=False, indent=2)
