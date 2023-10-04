import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.ndimage import generate_binary_structure
import common
from torchvision import transforms
import show_utils
import matplotlib.pyplot as plt

thresh1 = 0.3
thresh2 = 0.05

def get_bodyparts(heatmaps):
    all_bodyparts = []
    unique_id = 0

    for i in range(len(heatmaps)):
        filtered = maximum_filter(
            heatmaps[i], footprint=generate_binary_structure(2, 1)
        )
        peaks_coords = np.nonzero((filtered == heatmaps[i]) * (heatmaps[i] > thresh1))

        # if no peaks found
        if peaks_coords[0].size == 0:
            all_bodyparts.append([])

        else:
            my_list = []
            peaks_scores = heatmaps[i][peaks_coords]
            # use xy coords
            peaks_coords = np.flip(np.array(peaks_coords).T)

            for j in range(len(peaks_scores)):
                bodypart = {
                    "coords": peaks_coords[j],
                    "score": peaks_scores[j],
                    "part_id": i,
                    "id": unique_id,
                }
                unique_id += 1
                my_list.append(bodypart)
            all_bodyparts.append(my_list)

    return all_bodyparts


def get_limb_scores(pafs, bodyparts, image_size):
    all_limb_scores = []

    for i in range(len(pafs) // 2):
        partA_id = common.connect_skeleton[i][0]
        partB_id = common.connect_skeleton[i][1]

        partsA = bodyparts[partA_id]
        partsB = bodyparts[partB_id]

        pafx = pafs[2 * i]
        pafy = pafs[2 * i + 1]

        if partsA and partsB:
            my_list = []
            for pka in partsA:
                for pkb in partsB:
                    v = pkb["coords"] - pka["coords"]
                    v_magn = np.sqrt(v[0] ** 2 + v[1] ** 2) + 1e-8
                    v_norm = v / v_magn

                    # line
                    line_x = np.round(
                        np.linspace(pka["coords"][0], pkb["coords"][0], 10)
                    ).astype(int)
                    line_y = np.round(
                        np.linspace(pka["coords"][1], pkb["coords"][1], 10)
                    ).astype(int)

                    # flip indexing for ij coords
                    paf_x = pafx[line_y, line_x]
                    paf_y = pafy[line_y, line_x]
                    paf_vec = (paf_x, paf_y)

                    scores = np.dot(v_norm, paf_vec)
                    # penalize limbs longer than image.H/2
                    penalty = min(0.5 * image_size[1] / v_magn - 1, 0)
                    penalized_score = scores.mean() + penalty

                    criterion1 = np.count_nonzero(scores > thresh2) > 0.8 * len(scores) # here 0.8 is too stringent
                    criterion2 = penalized_score > 0

                    if criterion1 and criterion2:
                        limb = {
                            "part_a": pka,
                            "part_b": pkb,
                            "limb_score": penalized_score,
                            "limb_id": i,
                        }

                        my_list.append(limb)
            all_limb_scores.append(my_list)
        else:
            all_limb_scores.append([])

    return all_limb_scores


def get_connections(limb_scores, bodyparts):
    all_connections = []

    for i in range(len(limb_scores)):
        if limb_scores[i]:
            partA_id = common.connect_skeleton[i][0]
            partB_id = common.connect_skeleton[i][1]

            num_a = len(bodyparts[partA_id])
            num_b = len(bodyparts[partB_id])
            max_connections = min(num_a, num_b)

            limb_scores[i] = sorted(
                limb_scores[i], key=lambda x: x["limb_score"], reverse=True
            )

            my_list = []
            used = []
            for x in limb_scores[i]:
                if x["part_a"]["id"] not in used and x["part_b"]["id"] not in used:
                    my_list.append(x)
                    used.append(x["part_a"]["id"])
                    used.append(x["part_b"]["id"])

                    if len(my_list) >= max_connections:
                        break
            all_connections.append(my_list)
        else:
            all_connections.append([])

    return all_connections


def find_in_list(my_list, item):
    for index, x in enumerate(my_list):
        if item in x:
            return index
    return None  # Item not found


def group_limbs(connections):
    groups = []
    bins = []

    for x in connections:
        if x:
            for y in x:
                index_a = find_in_list(bins, y["part_a"]["id"])
                index_b = find_in_list(bins, y["part_b"]["id"])

                if index_a is None and index_b is None:
                    bins.append([y["part_a"]["id"], y["part_b"]["id"]])
                    groups.append([y])

                elif index_a is not None and index_b is None:
                    bins[index_a].append(y["part_b"]["id"])
                    groups[index_a].append(y)

                elif index_a is None and index_b is not None:
                    bins[index_b].append(y["part_a"]["id"])
                    groups[index_b].append(y)

                elif index_a is not None and index_b is not None:
                    if index_a == index_b:
                        groups[index_a].append(y)
                    else:
                        merged_bins = bins[index_a] + bins[index_b]
                        del bins[index_a]
                        del bins[index_b]
                        bins.append(merged_bins)

                        merged_groups = groups[index_a] + groups[index_b]
                        del groups[index_a]
                        del groups[index_b]
                        groups.append(merged_groups)
    return groups


def group_parts(groups):
    person_parts = []

    for x in groups:
        my_list = []
        for y in x:
            my_list.append(y["part_a"])
            my_list.append(y["part_b"])
        person_parts.append(my_list)

    unique = []
    for x in person_parts:
        my_list = []
        used = []
        for y in x:
            if y["id"] not in used:
                my_list.append(y)
                used.append(y["id"])
        unique.append(my_list)

    for i in range(len(unique)):
        unique[i] = sorted(unique[i], key=lambda x: x["part_id"])

    return unique


def post_process(image_size, heatmaps, pafs):
    heatmaps = transforms.functional.resize(
        heatmaps,
        (image_size[1], image_size[0]),
        transforms.functional.InterpolationMode.BICUBIC,
    )

    pafs = transforms.functional.resize(
        pafs,
        (image_size[1], image_size[0]),
        transforms.functional.InterpolationMode.NEAREST,
    )

    heatmaps = heatmaps.numpy()
    pafs = pafs.numpy()

    bodyparts = get_bodyparts(heatmaps)
    limb_scores = get_limb_scores(pafs, bodyparts, image_size)
    connections = get_connections(limb_scores, bodyparts)
    limb_groups = group_limbs(connections)
    part_groups = group_parts(limb_groups)

    return part_groups


def coco_format(part_groups):
    keypoints = []

    for x in part_groups:
        my_list = [None] * 17
        for y in x:
            my_list[y["part_id"]] = y["coords"].tolist()
        keypoints.append(my_list)

    return keypoints


def show_keypoints(image, keypoints):
    res = show_utils.draw_keypoints(
        image, keypoints, connectivity=common.connect_skeleton
    )
    plt.imshow(res)
