import numpy as np
import common
import torch
import torchvision.transforms.functional as F


def get_heatmaps(keypoints, size, visibility):
    parts = []
    for i in range(18):
        parts.append([x[i] for x in keypoints if x[i][2] in visibility])

    def gaussian(center, sigma=1, size=size):
        x, y = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        gaussian = np.exp(-((dist / sigma)**2))
        return gaussian

    heatmaps = []
    for x in parts:
        my_list = [gaussian((y[0], y[1])) for y in x]
        if my_list:
            heatmaps.append(np.maximum.reduce(my_list))
        else:
            heatmaps.append(np.zeros((size[1], size[0])))

    background = np.maximum((1 - np.maximum.reduce(heatmaps)), 0)
    heatmaps.append(background)
    return heatmaps


def person_paf(person, limb, size, visibility):
    part1_vis = person[limb[0]][2]
    part2_vis = person[limb[1]][2]
    if part1_vis and part2_vis in visibility:
        part1 = np.array([person[limb[0]][0], person[limb[0]][1]])
        part2 = np.array([person[limb[1]][0], person[limb[1]][1]])

        v = part2 - part1
        v_magn = np.sqrt(v[0]**2 + v[1]**2)
        if v_magn == 0:
            return (0, 0), np.zeros((size[1], size[0])).astype(int)
        v_norm = v / v_magn
        v_norm_orth = np.array([v_norm[1], -v_norm[0]])

        px, py = np.meshgrid(np.arange(size[0]), np.arange(size[1]))

        xp_x = px.flatten() - part1[0]
        xp_y = py.flatten() - part1[1]
        vec_xp = np.array([xp_x, xp_y])

        dot1 = np.dot(v_norm, vec_xp).reshape(size[1], size[0])
        cond1 = np.logical_and(dot1 >= 0, dot1 <= v_magn)

        s_thresh = 0.7
        dot2 = np.abs(np.dot(v_norm_orth, vec_xp).reshape(size[1], size[0]))
        cond2 = dot2 <= s_thresh

        paf_loc = np.logical_and(cond1, cond2)

        return v_norm, paf_loc
    else:
        return (0, 0), np.full((size[1], size[0]), False)


def get_pafs(keypoints, size, visibility):
    pafs = []
    for limb in common.connect_skeleton:
        res = [person_paf(x, limb, size, visibility) for x in keypoints]
        vectors, paf_locs = zip(*res)

        listx = []
        for v, loc in zip(vectors, paf_locs):
            arrx = np.zeros((size[1], size[0]))
            arrx[loc] = v[0]
            listx.append(arrx)

        listy = []
        for v, loc in zip(vectors, paf_locs):
            arry = np.zeros((size[1], size[0]))
            arry[loc] = v[1]
            listy.append(arry)

        num = np.add.reduce(paf_locs)
        num[num == 0] = 1
        paf_locs = np.maximum.reduce(paf_locs)

        paf_x = np.add.reduce(listx) / num
        paf_y = np.add.reduce(listy) / num

        pafs.append(paf_locs)
    return pafs


def get_mask_out(image, target, coco, size):
    masks = [coco.annToMask(x) for x in target if x["num_keypoints"] == 0]
    if masks:
        mask_out = np.add.reduce(masks)
        mask_out = np.where(mask_out >= 1, 0, 1)
    else:
        mask_out = np.ones((image.size[1], image.size[0]))
    mask_out = torch.tensor(mask_out, dtype=torch.float32)
    mask_out = F.resize(mask_out.unsqueeze_(0), size, F.InterpolationMode.NEAREST)
    return mask_out


def add_neck(keypoints, visibility):
    for i in range(len(keypoints)):
        l_should = np.asarray(keypoints[i][5])
        r_should = np.asarray(keypoints[i][6])

        neck = (l_should + r_should) / 2

        if l_should[2] in visibility and r_should[2] in visibility:
            neck[2] = 2
        else:
            neck[2] = l_should[2] * r_should[2]

        keypoints[i].append(neck.tolist())
    return keypoints
