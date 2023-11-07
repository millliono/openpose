import coco_dataset
import pathlib
from torchvision import transforms
import torch
import model
import post
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

inp_size = (368, 368)
targ_size = (46, 46)
coco_dataset = coco_dataset.CocoKeypoints(
    root=str(pathlib.Path("../coco") / "images" / "val2017"),
    annFile=str(pathlib.Path("../coco") / "annotations" / "annotations" / "person_keypoints_val2017.json"),
    transform=transforms.Compose([
        # transforms.Resize(inp_size),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]),
    targ_size=targ_size,
    train=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"  # comment when using modern gpu
if device == "cuda":
    device = "cuda:0"
    model = torch.nn.DataParallel(model.openpose(), device_ids=[0])
    model.load_state_dict(torch.load("save_modelGG.pth"))
else:
    model = model.openpose()
model.eval()

test_loader = DataLoader(
    dataset=coco_dataset,
    batch_size=1,
    num_workers=6,
    shuffle=False,
    drop_last=True,
)

loop = tqdm(test_loader, leave=True)
with torch.no_grad():
    my_list = []
    for batch_idx, (input_images, orig_sizes, ids) in enumerate(loop):
        input_images = input_images.to(device)
        pred_pafs, pred_htmps, _, _ = model(input_images)

        for paf, htmp, og_size, id in zip(pred_pafs.cpu(), pred_htmps.cpu(), orig_sizes, ids):
            kpt_groups = post.post_process(htmp, paf, og_size.tolist())
            if kpt_groups:
                keypoints = post.coco_format(kpt_groups)
                
                # delete neck keypoint
                for i in range(len(keypoints)):
                    keypoints[i].pop()
                    
                keypoints = np.array(keypoints).reshape(-1, 51)
                keypoints = keypoints.tolist()
                

                for x in keypoints:
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
    cocoGt = COCO(annFile)  # load annotations
    cocoDt = cocoGt.loadRes("predictions.json")  # load model outputs

    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, "keypoints")
    cocoEval.params.imgIds = coco_dataset.ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    # return Average Precision
    print(cocoEval.stats[0])
