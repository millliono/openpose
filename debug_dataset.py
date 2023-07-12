import coco_dataset
import pathlib
import show_utils
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import torch
import common
import dataset_utils


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

sample = coco_dataset[6]
image, pafs, heatmaps, keypoints = sample

show_utils.show_annotated(image, keypoints)

# plt.figure()
# show_utils.show_heatmaps_combined(heatmaps)

# plt.figure()
# show_utils.show_pafs_combined(pafs)

# plt.figure()
# show_utils.show_pafs_quiver_combined(pafs)


# show_utils.show_heatmaps(heatmaps)
# show_utils.show_pafs(pafs)

# show_utils.show_pafs_quiver(pafs)

plt.show()
