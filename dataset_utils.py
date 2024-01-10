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
    p1_vis = person[limb[0]][2]
    p2_vis = person[limb[1]][2]

    if (p1_vis and p2_vis) not in visibility:
        rep = np.zeros((size[1], size[0]))
        return rep, rep, rep

    p1 = np.array([person[limb[0]][0], person[limb[0]][1]])
    p2 = np.array([person[limb[1]][0], person[limb[1]][1]])

    v = p2 - p1
    v_norm = v / (np.sqrt(v[0]**2 + v[1]**2) + 1e-5)

    locs = np.zeros((size[1], size[0]))
    rr, cc, val = weighted_line(p1[1], p1[0], p2[1], p2[0], w=.1, rmin=0, rmax_x=size[0], rmax_y=size[1])
    locs[rr, cc] = 1

    pafx = np.where(locs == 1, v_norm[0], 0)
    pafy = np.where(locs == 1, v_norm[1], 0)

    return pafx, pafy, locs


def get_pafs(keypoints, size, visibility):
    pafs = []
    paf_locs = []
    for limb in common.connect_skeleton:
        res = [person_paf(x, limb, size, visibility) for x in keypoints]
        pafx, pafy, locs = zip(*res)

        locs = np.sum(locs, axis=0)
        pafx = np.sum(pafx, axis=0)
        pafy = np.sum(pafy, axis=0)

        paf_locs.append(locs)
        pafs.append(pafx)
        pafs.append(pafy)
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


def trapez(y, y0, w):
    return np.clip(np.minimum(y + 1 + w / 2 - y0, -y + 1 + w / 2 + y0), 0, 1)


def weighted_line(r0, c0, r1, c1, w, rmin, rmax_x, rmax_y):
    # The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
    # If either of these cases are violated, do some switches.
    if abs(c1 - c0) < abs(r1 - r0):
        # Switch x and y, and switch again when returning.
        xx, yy, val = weighted_line(c0, r0, c1, r1, w, rmin=rmin, rmax_x=rmax_y, rmax_y=rmax_x)
        return (yy, xx, val)

    # At this point we know that the distance in columns (x) is greater
    # than that in rows (y). Possibly one more switch if c0 > c1.
    if c0 > c1:
        return weighted_line(r1, c1, r0, c0, w, rmin=rmin, rmax_x=rmax_x, rmax_y=rmax_y)

    # The following is now always < 1 in abs
    slope = (r1 - r0) / (c1 - c0)

    # Adjust weight by the slope
    w *= np.sqrt(1 + np.abs(slope)) / 2

    # We write y as a function of x, because the slope is always <= 1
    # (in absolute value)
    x = np.arange(c0, c1 + 1, dtype=float)
    y = x * slope + (c1 * r0 - c0 * r1) / (c1 - c0)

    # Now instead of 2 values for y, we have 2*np.ceil(w/2).
    # All values are 1 except the upmost and bottommost.
    thickness = np.ceil(w / 2)
    yy = (np.floor(y).reshape(-1, 1) + np.arange(-thickness - 1, thickness + 2).reshape(1, -1))
    xx = np.repeat(x, yy.shape[1])
    vals = trapez(yy, y.reshape(-1, 1), w).flatten()

    yy = yy.flatten()

    # Exclude useless parts and those outside of the interval
    # to avoid parts outside of the picture
    mask = np.logical_and.reduce((yy >= rmin, yy < rmax_y, xx >= rmin, xx < rmax_x, vals > 0))

    return (yy[mask].astype(int), xx[mask].astype(int), vals[mask])
