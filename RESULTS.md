# Results

One experiment = one change. Every number is mAP@0.5 from `scripts/evaluate_map.py` on the same fixed validation set (300 real photos, `splits/val_image_ids.json`, seed 2). Changing the split invalidates every row.

| ID | What changed | mAP@0.5 | Time | Conclusion |
|---|---|---|---|---|
| A | Baseline: real photos only (823 train), yolov8n, imgsz 640, 50 epochs, Ultralytics defaults (`optimizer=auto`, resolved to AdamW lr 2.7e-05) | 0.024 (Ultralytics own: 0.0258) | ~58 min on Mac M4 | Undertrained, not overfitted: mAP50 rose almost linearly until epoch 50 and `cls_loss` was still falling, box and dfl losses were flat. AP@0.75 (0.022) is close to AP@0.5 (0.024), so boxes are placed well when found and classification is the bottleneck. Hypothesis: the automatic lr was too small, tested in A2. |
| A2 | A with the optimizer set explicitly: AdamW, lr0 0.001 (A used `optimizer=auto`, which resolved to AdamW 2.7e-05). Nothing else changed. | 0.455 (Ultralytics own: 0.4987) | ~64 min training, ~1 h 10 min wall clock, Mac M4 | The automatic learning rate was the bottleneck: one change gave about 19x higher mAP (`cls_loss` at epoch 19 was 1.89 instead of 5.06). mAP50 flattens over the last epochs while the lr decays to 3e-05. AP@0.75 (0.422) is close to AP@0.5 (0.455), boxes are placed well. About three quarters of the ~0.6 hackathon reference (from memory, unverified). |
| A3 | A2 with imgsz 960. Hypothesis: more pixels per bottle make labels readable and help classification. Runs on Kaggle (T4), the Mac does not have the memory; the experimental change is only imgsz, hardware and batch size are not part of the comparison. | | | |
| B | A2 plus synthetic images mixed into training | | | |
| C | Pretrain on synthetic, fine-tune on real, both with A2 settings | | | |

Notes:

- Ultralytics prints its own mAP50 during validation. It is only a cross-check, the table uses the `evaluate_map.py` number.
- `cache="ram"` can make runs slightly non-deterministic, so differences below about one point are noise.
