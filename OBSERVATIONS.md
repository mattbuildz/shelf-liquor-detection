# Observations and problems during training

A log of what went wrong or looked odd, with evidence and status, so I can answer "what problems did you have". Status is one of: **measured** (seen in a log or number), **hypothesis** (plausible, not tested), **resolved**.

## Speed

- **First training epoch was slow: ~6.3 s per batch of 16.** The run used `Using 0 dataloader workers` (Ultralytics sets this on MPS), so one thread decodes 12-megapixel JPEGs and builds mosaics. Fix: `cache="ram"` (images decoded once at 640 px). Result: ~0.8-1 s per batch, ~54 s per epoch, ~67 s with validation. Status: **resolved, measured**. Caveat: the first epoch of the slow run included warm-up, so the comparison is not perfectly clean.
- **Memory pressure on the Mac.** During run A the training process held ~20.7 GB of 24 GB and swap was ~8.3 GB. Epoch time stayed at ~54 s. GPU memory grew from 7.5 to 8.3 GB over the epochs. Consequence: imgsz 960 at batch 16 will not fit comfortably on the Mac, so larger runs go to Kaggle or use a smaller batch. Status: **measured**.

## Data

- **Duplicate ground-truth boxes.** Ultralytics removed 94 duplicate labels from the 300 validation photos (4097 -> 4003 boxes) and 85 from one training photo. `pycocotools` keeps them. See `LIMITATIONS.md`. Status: **measured**.

## Pipeline checks

- **0 predictions from the smoke model.** After 3 epochs on 10% of the data, `predict_val.py` returned 0 boxes at `conf=0.001`. Likely cause: all class scores below the threshold. Status: **hypothesis**.
- **Pipeline tested with the pretrained `yolov8n.pt`.** 89515 boxes for 300 images, `evaluate_map.py` ran without error, mAP 0.000 as expected (COCO classes do not match the 369 products). This proves the file format is readable, not that box coordinates and the `+1` class shift are right. That is only confirmed once a trained model gives a clearly non-zero mAP and it matches the Ultralytics validation mAP50. Status: **partly verified**.

## Run A (real photos only, imgsz 640, 50 epochs, Ultralytics defaults)

- **Low recall, learning slowly.** Epochs 15-19: mAP50 0.0079 -> 0.0106, precision ~0.48, recall 0.019 -> 0.026, `box_loss` ~1.25, `dfl_loss` ~1.05, `cls_loss` 5.26 -> 5.06 (it started at 6.6). Reading: the model roughly knows where bottles are (box losses low and flat) but has not learned which of the 369 products it sees (class loss high). A detection with the wrong class does not count, so recall stays low and mAP with it. My first reading was "mAP is low because the model draws few boxes". That is only partly right: precision and recall are reported at one confidence threshold, while mAP uses all boxes down to `conf=0.001`. The main cause is weak classification. Status: **measured** (numbers).
- **Final result of run A (epoch 50).** mAP@0.5 = 0.024 with `evaluate_map.py` (Ultralytics own: 0.0258, close, so the prediction format is correct), precision 0.51, recall 0.050, ~58 min. mAP50 rose almost linearly until the end and `cls_loss` kept falling (6.5 -> 4.1), `box_loss` (~1.1) and `dfl_loss` (~1.03) were flat from about epoch 15, so the model was undertrained, not overfitted. AP@0.75 (0.022) is close to AP@0.5 (0.024): when it finds an object the box is well placed, classification is the weak part. A small jump in `cls_loss` and a dip in mAP at epoch ~40 come from `close_mosaic=10` (mosaic augmentation switches off for the last 10 epochs). The learning rate in `results.csv` decays from 2.7e-05 to 8e-07 over the run. Status: **measured**, cause still a **hypothesis** (see below).
- **Possible cause: tiny automatic learning rate.** With `optimizer=auto` Ultralytics ignores `lr0=0.01` and picked `AdamW(lr=2.7e-05)`, about 370 times smaller. Likely too small to learn 369 classes in 50 epochs. Test: experiment A2 with `optimizer="AdamW"` and `lr0=0.001` set explicitly, everything else as in A. Result: mAP@0.5 went from 0.024 to 0.455 (`evaluate_map.py`; Ultralytics own 0.4987), recall from 0.05 to 0.46, `cls_loss` at epoch 19 from 5.06 to 1.89. One change, an order of magnitude effect, so the hypothesis holds. Status: **measured, confirmed**.
- **`evaluate_map.py` is lower than Ultralytics' own mAP50 by ~9% (0.455 vs 0.4987), in A by ~7%.** Likely contributors: the 94 duplicate ground-truth boxes (see `LIMITATIONS.md`), COCO evaluation capping detections at 100 per image while the model returns up to 300, and different interpolation of the precision-recall curve. Not investigated further. Status: **hypothesis**. The table uses the `evaluate_map.py` number.

## Run A2 (A with AdamW lr0 0.001, nothing else changed)

- **Final numbers (epoch 50).** mAP@0.5 = 0.455 with `evaluate_map.py` (Ultralytics own 0.4987, best epoch 49 with 0.5006), mAP50-95 = 0.391 (Ultralytics), precision 0.63, recall 0.46, `box_loss` 0.81, `cls_loss` 1.05. Training time ~64 min (~1 h 10 min wall clock). At epoch 19 mAP50 was already 0.295 against 0.0106 in A. Status: **measured**.
- **Not overfitted.** Validation `cls_loss` (1.20) is above the training one (1.05) but still fell in the last epochs (1.245 at epoch 45, 1.201 at epoch 50). A gap of ~0.15 is expected, since the model has seen the training photos. Overfitting would show as validation loss rising while training loss keeps falling. Status: **measured**.
- **Gains flatten at the end.** mAP50 moves from 0.483 (epoch 45) to 0.499 (epoch 50), while the learning rate decays to 3e-05. It is not clear whether the model is near its limit or only slowed by the decaying lr. Whether more epochs help is untested. Status: **hypothesis, open**.
- **The error is mostly classification, not localisation.** AP@0.75 (0.422) is close to AP@0.5 (0.455), so boxes are placed well when an object is found. Status: **measured**.

## Next experiment and Kaggle

- **Decision: A3 = A2 with imgsz 960.** Reason (hypothesis): a bottle is ~24 px wide at 640, so labels cannot be read, at 960 it is ~36 px. Other candidates were more epochs, a larger model, synthetic images (B) and no horizontal flip. Resolution goes first because it is the most plausible lever for reading labels. Status: **planned**.
- **Kaggle instead of the Mac for A3.** The Mac used ~20 GB of 24 GB at 640, and memory grows roughly with the square of the image side. Expected speed-up on a T4 is ~2-3x at 640 and more at 960 (my guess, not measured). Setup costs 30-60 min, so the Mac stays for small runs. Status: **planned**.
- **Kaggle dataset upload limit.** The upload form refused 3904 loose files ("Max files exceeded: 3904 of 1000"). Fix: one `.zip` of `data/train` (Kaggle unzips it). Dataset stays private because publication rules are unconfirmed. Status: **resolved**.

## Environment

- **`pip` and `uv` mixed.** I tried `pip` to update Ultralytics in a `uv` project. Environment and `uv.lock` both ended at 8.4.51, so nothing changed, but changing versions between runs would break comparability. Rule: package changes go through `uv` only. Status: **resolved**.
- **Python version.** The venv runs Python 3.14 while `pyproject.toml` allows 3.11+. An f-string with nested double quotes in `scripts/evaluation/build_val_ground_truth.py` works on 3.12+ only. Status: **open, minor**.
