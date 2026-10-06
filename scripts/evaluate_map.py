from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


gt = COCO("splits/val_ground_truth.json")
dt = gt.loadRes("splits/val_predictions.json") #change between splits/val_perfect_predictions to see what perfect score would look like and switch to splits/val_predictions to see how your model is doing
ev = COCOeval(gt, dt, "bbox")
ev.evaluate(); ev.accumulate(); ev.summarize()