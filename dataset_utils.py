import numpy as np
import common
import torch
import torchvision.transforms.functional as F


def get_heatmaps(keypoints, size, visibility):
    parts = []
    for i in range(len(common.coco_keypoints)):
        parts.append([x[i] for x in keypoints if x[i][2] in visibility])

    def gaussian(center, sigma=1, size=size):
        x, y = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
        dist = (x - center[0])**2 + (y - center[1])**2
        exponent = dist / (2.0 * sigma * sigma)
        mask = exponent <= 2.3
        gauss = np.exp(-exponent)
        gauss = mask * gauss
        return gauss

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
    
    if (part1_vis and part2_vis) not in visibility:
        return (0, 0), np.full((size[1], size[0]), False)
    else:
        part1 = np.array([person[limb[0]][0], person[limb[0]][1]])
        part2 = np.array([person[limb[1]][0], person[limb[1]][1]])

        v = part2 - part1
        v_magn = np.sqrt(v[0]**2 + v[1]**2) + 1e-5
        v_norm = v / v_magn
        v_norm_orth = np.array([v_norm[1], -v_norm[0]])

        px, py = np.meshgrid(np.arange(size[0]), np.arange(size[1]))

        xp_x = px.flatten() - part1[0]
        xp_y = py.flatten() - part1[1]
        vec_xp = np.array([xp_x, xp_y])

        dot1 = np.dot(v_norm, vec_xp).reshape(size[1], size[0])
        cond1 = np.logical_and(dot1 >= -(0.2 * v_magn), dot1 <= 1.2 * v_magn)

        s_thresh = 1
        dot2 = np.abs(np.dot(v_norm_orth, vec_xp).reshape(size[1], size[0]))
        cond2 = dot2 <= s_thresh

        location = np.logical_and(cond1, cond2)

        return v_norm, location


def get_pafs(keypoints, size, visibility):
    pafs = []
    paf_locs = []
    for limb in common.connect_skeleton:
        res = [person_paf(x, limb, size, visibility) for x in keypoints]
        vectors, locations = zip(*res)

        listx = []
        for v, loc in zip(vectors, locations):
            arrx = np.zeros((size[1], size[0]))
            arrx[loc] = v[0]
            listx.append(arrx)

        listy = []
        for v, loc in zip(vectors, locations):
            arry = np.zeros((size[1], size[0]))
            arry[loc] = v[1]
            listy.append(arry)

        num = np.sum(locations, axis=0)
        num[num == 0] = 1
        locations = np.maximum.reduce(locations)

        paf_x = np.sum(listx, axis=0) / num
        paf_y = np.sum(listy, axis=0) / num

        pafs.append(paf_x)
        pafs.append(paf_y)
        paf_locs.append(locations)
    return pafs, paf_locs


def get_mask_out(image_size, target, coco, size):
    masks = [coco.annToMask(x) for x in target if x["num_keypoints"] == 0]
    if masks:
        mask_out = np.maximum.reduce(masks)
        mask_out = 1 - mask_out
    else:
        mask_out = np.ones((image_size[1], image_size[0]))
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
