import coco_dataset
import pathlib
import torch
import numpy as np
import utils
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

root = pathlib.Path("coco")
coco_dataset = coco_dataset.CocoKeypoints(
    root=str(root / "images" / "train2017"),
    annFile=str(
        root / "annotations" / "annotations" / "person_keypoints_train2017.json"
    ),
    cat_nms="person",
)

sample = coco_dataset[0]
image, annotations = sample

num_annotations = len(annotations["keypoints"])
keypoints = np.array(annotations["keypoints"]).reshape(num_annotations, 17, 3)
keypoints = torch.from_numpy(keypoints)

res = utils.draw_keypoints(
    F.pil_to_tensor(image),
    keypoints,
    visibility=[1,2],
    connectivity=utils.connect_skeleton,
    keypoint_color="blue",
    line_color="yellow",
    radius=4,
    width=3,
)
utils.show(res)
plt.show()
