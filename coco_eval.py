import coco_dataset
import pathlib
import numpy as np
from tqdm import tqdm
import post
import json
from torchvision.transforms import v2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

inp_size = 368
coco_dataset = coco_dataset.CocoKeypoints(
    root=str(pathlib.Path("../coco") / "images" / "val2017"),
    annFile=str(pathlib.Path("../coco") / "annotations" / "annotations" / "person_keypoints_val2017.json"),
    transform=None)

my_list = []
for i in tqdm(range(len(coco_dataset.ids))):
    inp, pafs, heatmaps, paf_locs, anns, id = coco_dataset[i]
    inp = v2.ToPILImage()(inp)
    humans = post.post_process(heatmaps, pafs, inp.size)

    if not humans:
        continue
    
    coco_humans = post.coco_format(humans)
    coco_humans = np.array(coco_humans).reshape(-1, 51)
    coco_humans = coco_humans.tolist()

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
cocoEval.params.imgIds = coco_dataset.ids
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
# return Average Precision
print(cocoEval.stats[0])