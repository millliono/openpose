import coco_dataset
import pathlib
import numpy as np
from tqdm import tqdm
import post
import json
from torchvision.transforms import v2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import model
import torch

coco_dataset = coco_dataset.CocoKeypoints(
    root=str(pathlib.Path("../coco") / "images" / "val2017"),
    annFile=str(pathlib.Path("../coco") / "annotations" / "annotations" / "person_keypoints_val2017.json"),
    transform=None)

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"  # comment when using modern gpu
if device == "cuda":
    device = "cuda:0"
    model = torch.nn.DataParallel(model.openpose(), device_ids=[0])
    model.load_state_dict(torch.load("save_modelGG.pth"))
else:
    model = model.openpose()
model.eval()

with torch.no_grad():
    my_list = []
    for i in tqdm(range(len(coco_dataset.ids))):
        inp, pafs, heatmaps, paf_locs, anns, id = coco_dataset[i]
        inp_size = v2.ToPILImage()(inp).size

        inp = v2.Compose(
            [v2.ToImage(),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])(inp)

        pred_pafs, pred_heatmaps = model(inp.unsqueeze_(0).to(device))
        pred_pafs.squeeze_(0)
        pred_heatmaps.squeeze_(0)

        humans = post.post_process(pred_heatmaps.cpu(), pred_pafs.cpu(), inp_size)

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
cocoEval.params.imgIds = coco_dataset.ids
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
# return Average Precision
print(cocoEval.stats[0])
