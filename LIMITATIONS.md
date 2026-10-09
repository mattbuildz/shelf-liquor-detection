# Limitations (notes for the README)

Collected during the work so the README can state them honestly. Add new items as they come up.

## Measurement

- **Local validation only.** The hackathon leaderboard no longer accepts submissions. The only number is mAP@0.5 on my own validation split: 300 of the 1123 real training photos, `random.seed(2)`, frozen in `splits/val_image_ids.json`. The ~0.6 leaderboard score of the original team is from memory and unverified.
- **Validation covers ~237 of 369 classes.** 10 classes appear only in validation, 45 only in real training photos (not evaluated). Many classes have 1-2 validation boxes, so their AP is noisy.
- **17 classes have no training boxes at all**, ~70 appear only on synthetic images.
- **Duplicate ground-truth boxes.** 94 of the 4097 validation boxes (~2.3%) are exact duplicates. Ultralytics drops them (sees 4003), `pycocotools` keeps them. A model returns one box per object, so the achievable mAP is slightly below 1. Training data has duplicates too (e.g. 85 in one image). Only the `evaluate_map.py` number goes into the results table, not the mAP printed by Ultralytics.

## Data

- **Unlabeled products.** Only products from the 369 classes have boxes. Shelves also hold many others (e.g. Jack Daniel's and Jim Beam bottles in one example photo) with no box. The model learns them as background, and a detection on them counts as a false positive. This matches how the test set is scored, but it is a ceiling on precision.
- **Data is not part of the repo.** The dataset comes from the organizers (Asseco), publication rules are unconfirmed. Only a few example images with boxes may be shown. Files from the organizers (`taxonomy.json`, `test_images.json`) are theirs, and the repo license does not cover the data.
- **Synthetic images** (`SIDG_TRAIN`, `SIDG_SYNTH_TRAIN`, 2781 of 3904) are renders, not photos. The test set is real photos only.

## Scope

- **Solo redo, planned for 24 h net of work.** I stopped timing during the experiments and most likely went over it. Few experiments, no extensive hyperparameter search.
- **One run per variant.** No run was repeated, so the spread between runs is unknown. Differences below about one point are treated as noise.