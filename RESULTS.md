# Results

One experiment = one change. Every number is mAP@0.5 from `scripts/evaluate_map.py` on the same fixed validation set (300 real photos, `splits/val_image_ids.json`, seed 2). Changing the split invalidates every row.

| ID | What changed | mAP@0.5 | Time | Conclusion | Commit |
|---|---|---|---|---|---|
| A | Baseline: real photos only (823 train), yolov8n, imgsz 640, 50 epochs, Ultralytics defaults (`optimizer=auto`, resolved to AdamW lr 2.7e-05) | 0.024 (Ultralytics own: 0.0258) | ~58 min on Mac M4 | Undertrained, not overfitted: mAP50 rose almost linearly until epoch 50 and `cls_loss` was still falling, box and dfl losses were flat. AP@0.75 (0.022) is close to AP@0.5 (0.024), so boxes are placed well when found and classification is the bottleneck. Hypothesis: the automatic lr was too small, tested in A2. | pending |
| A-960 | A with imgsz 960 | | | | |
| B | A plus synthetic images mixed into training | | | | |
| C | Pretrain on synthetic, fine-tune on real | | | | |

Notes:

- Ultralytics prints its own mAP50 during validation. It is only a cross-check, the table uses the `evaluate_map.py` number.
- `cache="ram"` can make runs slightly non-deterministic, so differences below about one point are noise.
