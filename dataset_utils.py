import numpy as np
import common
import torch
import torchvision.transforms.functional as F

def get_heatmaps(keypoints, size, visibility):
    parts_coords = []
    for part in range(17):
        parts_coords.append(
            [person[part] for person in keypoints if person[part][2] in visibility]
        )

    def get_gaussian(center, sigma=16, size=size):
        x, y = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
        dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        gaussian = np.exp(-((dist / sigma) ** 2))
        return gaussian

    heatmaps = []
    for part in parts_coords:
        temp = [get_gaussian((kpt[0], kpt[1])) for kpt in part]
        if temp:
            heatmaps.append(np.maximum.reduce(temp))  # paper says take max
        else:
            heatmaps.append(np.zeros((size[1], size[0])))

    return heatmaps


def get_limb_pafs(person, visibility, limb, size):
    part1_vis = person[limb[0]][2]
    part2_vis = person[limb[1]][2]
    if part1_vis and part2_vis in visibility:
        part1 = np.array([person[limb[0]][0], person[limb[0]][1]])
        part2 = np.array([person[limb[1]][0], person[limb[1]][1]])
        v = part2 - part1
        v_magn = np.sqrt(v[0] ** 2 + v[1] ** 2) + 1e-8
        v_norm = v / v_magn

        px, py = np.meshgrid(np.arange(size[0]), np.arange(size[1]))

        xp_x = px.flatten() - part1[0]
        xp_y = py.flatten() - part1[1]
        vec_xp = [xp_x, xp_y]

        l_thresh = v_magn
        temp = np.dot(v_norm, vec_xp).reshape(size[1], size[0])
        cond1 = np.where((temp >= 0) & (temp <= l_thresh), 1, 0)

        s_thresh = 16
        v_norm_orth = [v_norm[1], -v_norm[0]]
        res = np.abs(np.dot(v_norm_orth, vec_xp).reshape(size[1], size[0]))
        cond2 = np.where(res <= s_thresh, 1, 0)

        paf_locs = np.where((cond1 > 0) & (cond2 > 0), 1, 0)

        pafs_x = np.where(paf_locs > 0, v_norm[0], 0)
        pafs_y = np.where(paf_locs > 0, v_norm[1], 0)

        return np.stack([pafs_x, pafs_y], axis=0), paf_locs
    else:
        return np.zeros((2, size[1], size[0])), np.zeros((size[1], size[0]))


def get_pafs(keypoints, size, visibility):
    pafs = []
    for limb in common.connect_skeleton:
        res = [get_limb_pafs(person, visibility, limb, size) for person in keypoints]
        limb_paf, paf_locs = zip(*res)

        paf_locs = np.add.reduce(paf_locs)
        paf_locs = np.where(paf_locs <= 1, 1, paf_locs)

        limb_paf = np.add.reduce(limb_paf)

        # take the avg paf per limb
        limb_paf = limb_paf / paf_locs

        pafs.append(limb_paf[0])
        pafs.append(limb_paf[1])
    return pafs


def get_mask_out(image, target, coco):
    masks = [coco.annToMask(x) for x in target if x["num_keypoints"] == 0]
    if masks:
        mask_out = np.add.reduce(masks)
        mask_out = np.where(mask_out >= 1, 0, 1)
    else:
        mask_out = np.ones((image.size[1], image.size[0]))
    mask_out = torch.tensor(mask_out, dtype=torch.float32)
    mask_out = F.resize(mask_out.unsqueeze_(0), (46, 46), F.InterpolationMode.NEAREST)
    return mask_out