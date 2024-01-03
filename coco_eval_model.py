import pathlib
from tqdm import tqdm
import post
import json
import model as mdl
import torch
from coco_dataset import CocoKeypoints
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


@torch.no_grad()
def coco_eval_model(dataset, model, device):
    model.eval()

    my_list = []
    for i in tqdm(range(len(dataset.ids))):
        inp, _, _, id = dataset[i]
        h, w = inp.size()[-2:]
        inp_size = (w, h)

        pred_pafs, pred_heatmaps = model(inp.unsqueeze_(0).to(device))

        humans = post.post_process(pred_heatmaps.squeeze_(0).cpu(), pred_pafs.squeeze_(0).cpu(), inp_size)

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
    cocoEval.params.imgIds = dataset.ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval.stats[0]


if __name__ == '__main__':

    saved = "save.pth"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model = mdl.openpose()
        model.to(device)
        model.load_state_dict(torch.load(saved))
    else:
        model = mdl.openpose()

    mAP_dataset = CocoKeypoints(
        root=str(pathlib.Path("../coco") / "images" / "val2017"),
        annFile=str(pathlib.Path("../coco") / "annotations" / "annotations" / "person_keypoints_val2017.json"),
        transform=None)

    coco_eval_model(mAP_dataset, model, device)
