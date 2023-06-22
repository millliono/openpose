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

sample = coco_dataset[6]
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
# plt.show()

keypoints = keypoints.tolist()


def get_heatmaps(keypoints, visibility=[1, 2]):
    parts_coords = []
    for part in range(17):
        parts_coords.append(
            [person[part] for person in keypoints if person[part][2] in visibility]
        )

    def get_gaussian(center, sigma, size=224):
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        gaussian = np.exp(-((dist / sigma) ** 2))
        return gaussian

    heatmaps = []
    for t in parts_coords:
        temp = [get_gaussian((kpt[0], kpt[1]), sigma=1) for kpt in t]
        if temp:
            heatmaps.append(np.maximum.reduce(temp))
        else:
            heatmaps.append(np.zeros((224,224)))

    return heatmaps
    # utils.show2(heatmaps)
    # plt.show()
gg = get_heatmaps(keypoints)



def get_paf(person, size=224):
    part1_vis = person[limb[0]][2]
    part2_vis = person[limb[1]][2]
    if part1_vis and part2_vis in visibility:
        part1 = np.array([person[limb[0]][0], person[limb[0]][1]])
        part2 = np.array([person[limb[1]][0], person[limb[1]][1]])
        v = part2 - part1
        v_magn = np.sqrt(v[0] ** 2 + v[1] ** 2) + 1e-8
        v_norm = v / v_magn

        px, py = np.meshgrid(np.arange(size), np.arange(size))

        pixel_x = px.flatten() - part1[0]
        pixel_y = py.flatten() - part1[1]
        vec_xp = [pixel_x, pixel_y]

        temp = np.dot(v_norm, vec_xp).reshape(size, size)
        l_thresh = v_magn
        cond1 = np.where((temp >= 0) & (temp <= l_thresh), temp, 0)

        v_norm_orth = [v_norm[1], -v_norm[0]]
        res = np.abs(np.dot(v_norm_orth, vec_xp).reshape(size, size))
        cond2 = np.where((res >= 0) & (res <= l_thresh), res, 0)

        cond3 = np.where((cond1 > 0) & (cond2 > 0), 1, 0)

    return cond3


visibility = [1, 2]
pafs = []
for limb in utils.connect_skeleton:
    ans = [
        get_paf(person)
        for person in keypoints
        if person[limb[0]][2] in visibility and person[limb[1]][2] in visibility
    ]
    paf = np.add.reduce(ans)
    pafs.append(paf)

print("hi")