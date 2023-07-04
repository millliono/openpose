import coco_dataset
import pathlib
import show_utils
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import torch
import common


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

show_utils.show_annotated(image, keypoints)

def get_heatmaps(keypoints, visibility=[1, 2]):
    parts_coords = []
    for part in range(17):
        parts_coords.append(
            [person[part] for person in keypoints if person[part][2] in visibility]
        )

    def get_gaussian(center, sigma=1, size=224):
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        gaussian = np.exp(-((dist / sigma) ** 2))
        return gaussian

    heatmaps = []
    for part in parts_coords:
        temp = [get_gaussian((kpt[0], kpt[1])) for kpt in part]
        if temp:
            heatmaps.append(np.maximum.reduce(temp)) # paper says take max
        else:
            heatmaps.append(np.zeros((224, 224)))

    return heatmaps


def get_limb_pafs(person, visibility, limb, size=224):
    part1_vis = person[limb[0]][2]
    part2_vis = person[limb[1]][2]
    if part1_vis and part2_vis in visibility:
        part1 = np.array([person[limb[0]][0], person[limb[0]][1]])
        part2 = np.array([person[limb[1]][0], person[limb[1]][1]])
        v = part2 - part1
        v_magn = np.sqrt(v[0] ** 2 + v[1] ** 2) + 1e-8
        v_norm = v / v_magn

        px, py = np.meshgrid(np.arange(size), np.arange(size))

        xp_x = px.flatten() - part1[0]
        xp_y = py.flatten() - part1[1]
        vec_xp = [xp_x, xp_y]

        l_thresh = v_magn
        temp = np.dot(v_norm, vec_xp).reshape(size, size)
        cond1 = np.where((temp >= 0) & (temp <= l_thresh), 1, 0)

        s_thresh = 2 # TODO: find correct parameter
        v_norm_orth = [v_norm[1], -v_norm[0]]
        res = np.abs(np.dot(v_norm_orth, vec_xp).reshape(size, size))
        cond2 = np.where(res <= s_thresh, 1, 0)

        paf_locs = np.where((cond1 > 0) & (cond2 > 0), 1, 0)

        pafs_x = np.where(paf_locs > 0, v_norm[0], 0)
        pafs_y = np.where(paf_locs > 0, v_norm[1], 0)

        return np.stack([pafs_x, pafs_y], axis=0)
    else:
        return np.zeros((2, 224, 224))


def get_pafs(keypoints, visibility=[1,2]):
    pafs = []
    for limb in common.connect_skeleton:
        limb_paf = [get_limb_pafs(person, visibility, limb) for person in keypoints]
        limb_paf = np.add.reduce(limb_paf) # TODO: paper says take the average
        pafs.append(limb_paf)
    return pafs

pafs = get_pafs(keypoints)
heatmaps = get_heatmaps(keypoints)

plt.figure()
show_utils.show_pafs_combined(pafs)
plt.figure()
show_utils.show_heatmaps_combined(heatmaps)
plt.figure()
show_utils.show_pafs_quiver_combined(pafs)
# show_utils.show_pafs(pafs)
# show_utils.show_heatmaps(heatmaps)

plt.show()
