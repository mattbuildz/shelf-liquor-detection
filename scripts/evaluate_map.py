from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


gt = COCO("splits/val_ground_truth.json")
dt = gt.loadRes("splits/val_perfect_predictions.json")
ev = COCOeval(gt, dt, "bbox")
ev.evaluate(); ev.accumulate(); ev.summarize()