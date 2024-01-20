import coco_dataset
import pathlib
from tqdm import tqdm
import post
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

mAP_data = coco_dataset.CocoKeypoints(
    root=str(pathlib.Path("../coco") / "images" / "val2017"),
    annFile=str(pathlib.Path("../coco") / "annotations" / "annotations" / "person_keypoints_val2017.json"),
    transform=None)

my_list = []
for i in tqdm(range(len(mAP_data.ids))):
    inp, pafs, heatmaps, id, _, _ = mAP_data[i]
    h, w = inp.size()[-2:]
    inp_size = (w, h)

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
cocoEval.params.imgIds = mAP_data.ids
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

