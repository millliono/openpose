import coco_dataset
import pathlib
import utils
import matplotlib.pyplot as plt
from torchvision import transforms


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
