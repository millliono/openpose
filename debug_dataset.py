import coco_dataset
import pathlib
import torch
import numpy as np
import utils
import matplotlib.pyplot as plt
from torchvision import transforms


image_transform = torch.nn.Sequential(
    transforms.Resize([224, 224]),
)

root = pathlib.Path("coco")
coco_dataset = coco_dataset.CocoKeypoints(
    root=str(root / "images" / "train2017"),
    annFile=str(
        root / "annotations" / "annotations" / "person_keypoints_train2017.json"
    ),
    transform=image_transform,
    resize_keypoints=[224, 224],
)

sample = coco_dataset[0]
image, keypoints = sample

res = utils.draw_keypoints(
    transforms.functional.pil_to_tensor(image),
    keypoints,
    visibility=[1, 2],
    connectivity=utils.connect_skeleton,
    keypoint_color="blue",
    line_color="yellow",
    radius=2,
    width=2,
)
utils.show(res)
plt.show()
