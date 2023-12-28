import coco_dataset
import pathlib
import numpy as np
from tqdm import tqdm
import post
import json
from torchvision.transforms import v2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

coco_data = coco_dataset.CocoKeypoints(
    root=str(pathlib.Path("../coco") / "images" / "val2017"),
    annFile=str(pathlib.Path("../coco") / "annotations" / "annotations" / "person_keypoints_val2017.json"),
    transform=None)

my_list = []
for i in tqdm(range(len(coco_data.ids))):
    inp, pafs, heatmaps, paf_locs, anns, id = coco_data[i]
    inp_size = v2.ToPILImage()(inp).size
    humans = post.post_process(heatmaps, pafs, inp_size)

    if not humans:
        continue

    coco_humans = post.coco_format(humans)
    for x in coco_humans:
        person = {
            "image_id": id,
            "category_id": 1,
            "keypoints": x,
            "score": 1,
        }
        my_list.append(person)

with open("predictions.json", "w") as f:
    json.dump(my_list, f)

annFile = str(pathlib.Path("../coco") / "annotations" / "annotations" / "person_keypoints_val2017.json")
cocoGt = COCO(annFile)
cocoDt = cocoGt.loadRes("predictions.json")

cocoEval = COCOeval(cocoGt, cocoDt, "keypoints")
cocoEval.params.imgIds = coco_data.ids
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
# return Average Precision
print(cocoEval.stats[0])
