import coco_dataset
import pathlib
import utils
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import torch


coco_dataset = coco_dataset.CocoKeypoints(
    root=str(pathlib.Path("coco") / "images" / "train2017"),
    annFile=str(
        pathlib.Path("coco")
        / "annotations"
        / "annotations"
        / "person_keypoints_train2017.json"
    ),
    transform=transforms.Resize([224, 224]),
    resize_keypoints_to=[224, 224],
)

sample = coco_dataset[2]
image, keypoints = sample

# res = utils.draw_keypoints(
#     transforms.functional.pil_to_tensor(image),
#     torch.from_numpy(keypoints),
#     visibility=[1, 2],
#     connectivity=utils.connect_skeleton,
#     keypoint_color="blue",
#     line_color="yellow",
#     radius=2,
#     width=2,
# )
# utils.show1(res)




keypoints = keypoints.tolist()
parts_coords = []
for part in range(17):
    parts_coords.append([person[part] for person in keypoints])


def get_gaussian(center, sigma, size=224):
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    gaussian = np.exp(-((dist / sigma) ** 2))
    return gaussian

heatmaps = []
for t in parts_coords:
    single = [get_gaussian((kpt[0], kpt[1]), sigma=1) for kpt in t]
    heatmaps.append(np.maximum.reduce(single))

utils.show2(heatmaps)
plt.show()