import numpy as np
import common


def tf_resize_keypoints(keypoints, orig_size=[224, 224], new_size=[28, 28]):
    # this is a transform that resizes keypoints after an image resizing transform
    target_size = new_size
    width_resize = target_size[0] / orig_size[0]
    height_resize = target_size[1] / orig_size[1]
    resized_keypoints = keypoints * np.array([width_resize, height_resize, 1])
    return resized_keypoints


def get_heatmaps(keypoints, visibility=[1, 2], size=28):
    keypoints = tf_resize_keypoints(keypoints, new_size=[size, size])

    parts_coords = []
    for part in range(17):
        parts_coords.append(
            [person[part] for person in keypoints if person[part][2] in visibility]
        )

    def get_gaussian(center, sigma=1, size=size):
        x, y = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")  # IJ
        dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        gaussian = np.exp(-((dist / sigma) ** 2))
        return gaussian

    heatmaps = []
    for part in parts_coords:
        temp = [get_gaussian(np.flip((kpt[0], kpt[1]))) for kpt in part]  # FLIP HERE
        if temp:
            heatmaps.append(np.maximum.reduce(temp))  # paper says take max
        else:
            heatmaps.append(np.zeros((size, size)))

    return heatmaps


def get_limb_pafs(person, visibility, limb, size=28):
    part1_vis = person[limb[0]][2]
    part2_vis = person[limb[1]][2]
    if part1_vis and part2_vis in visibility:
        part1 = np.flip([person[limb[0]][0], person[limb[0]][1]])  # FLIP HERE
        part2 = np.flip([person[limb[1]][0], person[limb[1]][1]])  # FLIP HERE
        v = part2 - part1
        v_magn = np.sqrt(v[0] ** 2 + v[1] ** 2) + 1e-8
        v_norm = v / v_magn

        px, py = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")  # IJ

        xp_x = px.flatten() - part1[0]
        xp_y = py.flatten() - part1[1]
        vec_xp = [xp_x, xp_y]

        l_thresh = v_magn
        temp = np.dot(v_norm, vec_xp).reshape(size, size)
        cond1 = np.where((temp >= 0) & (temp <= l_thresh), 1, 0)

        s_thresh = 1  # TODO: find correct parameter
        v_norm_orth = [v_norm[1], -v_norm[0]]
        res = np.abs(np.dot(v_norm_orth, vec_xp).reshape(size, size))
        cond2 = np.where(res <= s_thresh, 1, 0)

        paf_locs = np.where((cond1 > 0) & (cond2 > 0), 1, 0)

        pafs_x = np.where(paf_locs > 0, v_norm[0], 0)
        pafs_y = np.where(paf_locs > 0, v_norm[1], 0)

        return np.stack([pafs_x, pafs_y], axis=0)
    else:
        return np.zeros((2, size, size))


def get_pafs(keypoints, visibility=[1, 2]):
    keypoints = tf_resize_keypoints(keypoints)

    pafs = []
    for limb in common.connect_skeleton:
        limb_paf = [get_limb_pafs(person, visibility, limb) for person in keypoints]
        limb_paf = np.add.reduce(limb_paf)  # TODO: paper says take the average
        pafs.append(limb_paf[0])
        pafs.append(limb_paf[1])
    return pafs
