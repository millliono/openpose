import coco_dataset
import pathlib
from tqdm import tqdm
import post
import json
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import model
import torch


def coco_eval_model(dataloader, model, device):
    model.eval()

    loop = tqdm(dataloader, leave=True)
    with torch.no_grad():
        my_list = []
        for i, (input, pafs, heatmaps, id) in enumerate(loop):

            pred_pafs, pred_heatmaps = model(input.to(device))
            pred_pafs.squeeze_(0)
            pred_heatmaps.squeeze_(0)

            h, w = input.size()[-2:]
            inp_size = (w, h)
            humans = post.post_process(pred_heatmaps.cpu(), pred_pafs.cpu(), inp_size)

            if not humans:
                continue

            coco_humans = post.coco_format(humans)
            for x in coco_humans:
                person = {
                    "image_id": id.item(),
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
    # return Average Precision
    print(cocoEval.stats[0])
    return cocoEval.stats[0]


if __name__ == '__main__':

    saved = "saved.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        device = "cuda:3"
        model = torch.nn.DataParallel(model.openpose(), device_ids=[3])
        model.load_state_dict(torch.load(saved))
    else:
        model = model.openpose()

    mAP_data = coco_dataset.CocoKeypoints(
        root=str(pathlib.Path("../coco") / "images" / "val2017"),
        annFile=str(pathlib.Path("../coco") / "annotations" / "annotations" / "person_keypoints_val2017.json"),
        transform=None)

    mAP_loader = DataLoader(dataset=mAP_data, batch_size=1)

    coco_eval_model(mAP_loader, model, device)
