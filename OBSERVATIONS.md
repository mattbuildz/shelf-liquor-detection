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
- **Possible cause: tiny automatic learning rate.** With `optimizer=auto` Ultralytics ignores `lr0=0.01` and picked `AdamW(lr=2.7e-05)`, about 370 times smaller. Likely too small to learn 369 classes in 50 epochs. Test: experiment A2 with `optimizer="AdamW"` and `lr0=0.001` set explicitly, everything else as in A. Status: **hypothesis**.

## Environment

- **`pip` and `uv` mixed.** I tried `pip` to update Ultralytics in a `uv` project. Environment and `uv.lock` both ended at 8.4.51, so nothing changed, but changing versions between runs would break comparability. Rule: package changes go through `uv` only. Status: **resolved**.
- **Python version.** The venv runs Python 3.14 while `pyproject.toml` allows 3.11+. An f-string with nested double quotes in `scripts/evaluation/build_val_ground_truth.py` works on 3.12+ only. Status: **open, minor**.
