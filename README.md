# Shelf product detection, Hackology II solo redo

Object detection on photos of store shelves. There are 369 classes and each class is one specific product. The metric is mAP@0.5 and the annotations use the COCO format.

This repository is my solo redo of the Hackology II hackathon task, which ran on 23 and 24 May 2026. I am doing it again alone, with a budget of 24 hours of net work, so that I can explain every step. The score matters less than understanding where it comes from.

## Current state

The best run so far is A2 with mAP@0.5 of 0.455. I measured it on 300 real photos held out from training, using my own validation split. It is a local number and not a leaderboard score. The hackathon leaderboard no longer accepts submissions.

Three more experiments are planned and have no result yet. `RESULTS.md` is the source of truth for every score in this README.

## Task and data

- The training set has 3904 images. 1123 are real photos and 2781 are synthetic renders.
- Each image has a `source_dataset` field in `annotations.json`. The values `SIDG_TRAIN` and `SIDG_SYNTH_TRAIN` mark synthetic images.
- The test set contains real photos only and has no annotations.
- Only products from the 369 classes have boxes. Other products on the same shelf have none, so the model learns them as background.

The data is not in this repository. It comes from the organizers and the rules for publishing it are not confirmed. `taxonomy.json` and `test_images.json` are the organizers' files.

## Method

### Validation split

I took 300 of the 1123 real photos as the validation set, with `random.seed(2)`. The ids are frozen in `splits/val_image_ids.json`, which is tracked in git. The other 823 real photos are the training set for runs A and A2.

The split is frozen because a score only means something next to another score on the same photos. Changing the split invalidates every row in the results table. Validation uses real photos only because the test set is real photos only.

The 300 validation photos carry 4097 ground-truth boxes.

### Pipeline

1. `scripts/validation/data_split.py` samples the 300 validation ids and saves them.
2. `scripts/evaluation/build_val_ground_truth.py` cuts the COCO annotations down to those 300 photos and writes the ground truth for evaluation.
3. `scripts/training/convert_coco_to_yolo.py` writes the train and validation image lists and one YOLO label file per photo. COCO boxes are `[x, y, width, height]` in pixels from the top-left corner. YOLO wants the box center and size as fractions of the image size. COCO class ids run from 1 to 369 and YOLO indices from 0 to 368, so the script subtracts 1. Validation photos are excluded from the training list.
4. `scripts/training/make_dataset_yaml.py` writes the dataset config that Ultralytics reads.
5. `scripts/training/train.py` fine-tunes the model.
6. `scripts/training/predict_val.py` loads `best.pt` from the run folder, predicts on the 300 validation photos and converts the output back to COCO format. It adds 1 to the class index and turns corner coordinates into `[x, y, width, height]`.
7. `scripts/evaluate_map.py` scores the predictions against the ground truth with `pycocotools`.

### Training setup

| Setting | Value |
|---|---|
| Model | YOLOv8n, pretrained weights `yolov8n.pt` |
| Library | Ultralytics 8.4.51 |
| Training data | 823 real photos |
| Image size | 640 |
| Batch size | 16 |
| Epochs | 50 |
| Optimizer in run A | `optimizer=auto`, which resolved to AdamW with learning rate 2.7e-05 |
| Optimizer in run A2 | AdamW with `lr0=0.001`, set explicitly |
| Other | `cache="ram"`, `device="mps"` |
| Hardware | Mac M4, 24 GB RAM |
| Time per run | about 58 minutes for A, about 64 minutes for A2 |

### How the score is computed

`predict_val.py` runs the model with a confidence threshold of 0.001. mAP is computed from the whole ranking of detections, so the low-confidence boxes are needed too.

`evaluate_map.py` runs `COCOeval` from `pycocotools`. The number I report is the AP at IoU 0.50 from its summary.

Ultralytics prints its own mAP50 during validation. It gave 0.0258 for run A and 0.4987 for run A2. I use it as a cross-check only. The results table uses the `evaluate_map.py` number.

## Experiments

One experiment is one change. Every row is scored on the same 300 validation photos.

| ID | What changed | mAP@0.5 | Time |
|---|---|---|---|
| A | Baseline. Real photos only, YOLOv8n, image size 640, 50 epochs, Ultralytics defaults | 0.024 | about 58 min |
| A2 | A with the optimizer set explicitly to AdamW, `lr0=0.001` | 0.455 | about 64 min |
| A3 | A2 with image size 960, on a Kaggle T4 | planned | |
| B | A2 plus synthetic images mixed into training | planned | |
| C | Pretrain on synthetic images, fine-tune on real photos | planned | |

`cache="ram"` can make runs slightly non-deterministic, so differences below about one point are noise.

### What run A showed

Run A ended at 0.024. The model was undertrained, not overfitted. Its mAP50 rose almost linearly until epoch 50 and the class loss was still falling, from 6.5 to 4.1. The box loss and the dfl loss were flat from about epoch 15.

AP at IoU 0.75 was 0.022, close to the 0.024 at IoU 0.5. When the model found an object, the box was well placed. Classification was the bottleneck.

The cause was the learning rate. With `optimizer=auto` Ultralytics ignores `lr0=0.01` and picked AdamW with a learning rate of 2.7e-05, about 370 times smaller.

### What run A2 showed

Run A2 changed only the optimizer setting. mAP@0.5 went from 0.024 to 0.455, about 19 times higher. Recall went from 0.05 to 0.46. The class loss at epoch 19 was 1.89 instead of 5.06.

- Final precision was 0.63 and recall 0.46, both from the Ultralytics validation pass.
- AP at IoU 0.75 is 0.422, close to 0.455. The remaining error is mostly classification, not box placement.
- The run is not overfitted. Validation class loss is 1.20 against 1.05 in training, and it was still falling in the last epochs.
- Gains flatten at the end while the learning rate decays to 3e-05. Whether more epochs would help is untested.

### Why A3 is next

At image size 640 a bottle is about 24 pixels wide. At 960 it is about 36 pixels. My hypothesis is that more pixels per bottle make labels readable and help classification. The Mac does not have the memory for this run, so A3 goes to Kaggle. The experimental change is the image size only.

## Problems observed

The full log with evidence is in `OBSERVATIONS.md`. The ones worth knowing:

- **Learning rate chosen automatically.** Described above. Status: measured and confirmed by run A2.
- **Slow first epoch.** About 6.3 seconds per batch of 16. Ultralytics uses 0 dataloader workers on MPS, so one thread decoded 12-megapixel JPEGs and built mosaics. With `cache="ram"` a batch takes about 0.8 to 1 second. Status: resolved.
- **Memory pressure on the Mac.** During run A the training process held about 20.7 GB of 24 GB and swap was about 8.3 GB. Image size 960 at batch 16 will not fit comfortably. Status: measured.
- **Duplicate ground-truth boxes.** 94 of the 4097 validation boxes are exact duplicates. Ultralytics drops them and `pycocotools` keeps them. Status: measured.
- **Two evaluators, two numbers.** `evaluate_map.py` is lower than the Ultralytics mAP50 by about 9% in A2 and about 7% in A. Likely contributors are the duplicate boxes, the COCO limit of 100 detections per image and a different interpolation of the precision-recall curve. Status: hypothesis, not investigated.
- **Zero predictions from a smoke model.** After 3 epochs on 10% of the data, `predict_val.py` returned 0 boxes at confidence 0.001. Status: hypothesis.
- **Pipeline check with the pretrained model.** `yolov8n.pt` without fine-tuning gave 89515 boxes for the 300 photos and mAP 0.000, as expected, because COCO classes do not match the 369 products. That proved the file format. The coordinates and the class shift were confirmed later, when run A gave a non-zero score close to the Ultralytics one.
- **Python version.** `pyproject.toml` allows Python 3.11 and newer. `build_val_ground_truth.py` uses an f-string with nested double quotes, which works on 3.12 and newer only. My environment runs 3.14. Status: open, minor.

## Limitations

- **Local validation only.** The only number is mAP@0.5 on my own split. The original team's leaderboard score of about 0.6 is from memory and unverified, so I do not use it as a baseline.
- **Validation covers about 237 of 369 classes.** 10 classes appear only in validation. 45 appear only in real training photos and are not evaluated. Many classes have 1 or 2 validation boxes, so their AP is noisy.
- **17 classes have no training boxes at all.** About 70 appear only on synthetic images.
- **Duplicates lower the ceiling.** A model returns one box per object, so with duplicate ground truth the achievable mAP is slightly below 1.
- **Unlabeled products.** A detection on a product outside the 369 classes counts as a false positive. This matches how the test set is scored, but it limits precision.
- **Synthetic images are renders, not photos.** Runs A and A2 do not use them.
- **Few experiments.** This is a solo redo with 24 hours of net work. There is no extensive hyperparameter search.

`LIMITATIONS.md` has the longer version.

## How to reproduce

You need the dataset under `data/train/`. It is not part of this repository.

```bash
uv sync
```

Run the scripts from the repository root, in this order:

```bash
uv run python scripts/evaluation/build_val_ground_truth.py
uv run python scripts/training/convert_coco_to_yolo.py
uv run python scripts/training/make_dataset_yaml.py
uv run python scripts/training/train.py
uv run python scripts/training/predict_val.py
uv run python scripts/evaluate_map.py
```

- `splits/val_image_ids.json` is already in the repository. Do not run `data_split.py` again unless you mean to change the split.
- `train.py` uses `device="mps"`. Change it on other hardware. The run name and the settings are constants at the top of the file.
- `predict_val.py` has the path to the weights as a constant. Point it at the run you want to score.
- `evaluate_map.py` prints the COCO summary. The second line is the AP at IoU 0.50.

Three helper scripts are not part of the main path. `scripts/evaluation/make_perfect_predictions.py` builds predictions from the ground truth, to see what the evaluator prints for a perfect answer. `scripts/training/draw_labels.py` draws the YOLO labels of one photo to check the conversion. `scripts/validation/val_image_ids.py` prints how many classes appear in each part of the split.

## Repository layout

```
RESULTS.md               results table, source of truth for scores
OBSERVATIONS.md          log of problems with evidence and status
LIMITATIONS.md           limits of the measurement and the data
scripts/validation/      validation split and class coverage check
scripts/training/        label conversion, dataset config, training, prediction on validation
scripts/evaluation/      ground truth for the validation photos, perfect-answer check
scripts/evaluate_map.py  mAP with pycocotools
splits/val_image_ids.json  the frozen 300 validation ids
predict.py               organizers' prediction CLI, see below
notebooks/               notebooks from the organizers' template
taxonomy.json, test_images.json, download_data.sh, checksums.sha256   organizers' files
docs/organizers/README.md  original hackathon README, in Polish
ONBOARDING.md            organizers' onboarding guide, in Polish
```

`predict.py` is the command line interface the organizers required. It is still their baseline. By default it loads the COCO-pretrained `yolov8n.pt` and is not yet connected to the weights trained here.

## Authorship and AI assistance

Parts of the pipeline were written with AI assistance, at my explicit request, to save time:

- `group_boxes_by_image`, `coco_box_to_yolo_line` and `write_yolo_labels` in `scripts/training/convert_coco_to_yolo.py`
- `scripts/training/make_dataset_yaml.py`
- `scripts/training/train.py`
- `scripts/training/predict_val.py`

The rest of the code under `scripts/` is mine. `predict.py` and the notebooks come from the organizers' template. This README was drafted by an AI agent at my request, from `RESULTS.md`, `OBSERVATIONS.md` and `LIMITATIONS.md`.

## Origin and license

The repository started as the organizers' template for Hackology II. Their original instructions are in `docs/organizers/README.md` and `ONBOARDING.md`.

The code is under the GNU Affero General Public License v3, see `LICENSE`. The license does not cover the organizers' data files.
