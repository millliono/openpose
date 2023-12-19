import numpy as np
from scipy.ndimage import maximum_filter
from scipy.ndimage import generate_binary_structure
import common
from torchvision import transforms


def get_bodyparts(heatmaps):
    all_bodyparts = []
    unique_id = 0

    for i in range(len(heatmaps) - 1):  # exclude background heatmap
        filtered = maximum_filter(heatmaps[i], footprint=generate_binary_structure(2, 1))
        peaks_coords = np.nonzero((filtered == heatmaps[i]) * (heatmaps[i] > 0.1))

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
                    "part_type": i,
                    "id": unique_id,
                }
                unique_id += 1
                my_list.append(bodypart)
            all_bodyparts.append(my_list)

    return all_bodyparts


def get_limbs(pafs, bodyparts, image_size):
    valid_limbs = []

    for i in range(len(pafs) // 2):
        partA_id = common.connect_skeleton[i][0]
        partB_id = common.connect_skeleton[i][1]

        partsA = bodyparts[partA_id]
        partsB = bodyparts[partB_id]

        if not (partsA and partsB):
            continue

        pafx = pafs[2 * i]
        pafy = pafs[2 * i + 1]

        possible_limbs = []  # stores all possible limb connections
        for pka in partsA:
            for pkb in partsB:
                v = pkb["coords"] - pka["coords"]
                v_magn = np.sqrt(v[0]**2 + v[1]**2) + 1e-5
                v_norm = v / v_magn

                # line
                line_x = np.round(np.linspace(pka["coords"][0], pkb["coords"][0], 10)).astype(int)
                line_y = np.round(np.linspace(pka["coords"][1], pkb["coords"][1], 10)).astype(int)

                # flip indexing for ij coords
                paf_x = pafx[line_y, line_x]
                paf_y = pafy[line_y, line_x]
                paf_vec = (paf_x, paf_y)

                scores = np.dot(v_norm, paf_vec)
                # penalize limbs longer than image.H/2
                penalty = min(0.5 * image_size[1] / v_magn - 1, 0)
                penalized_score = scores.mean() + penalty

                criterion1 = np.count_nonzero(scores > 0.05) > 0.8 * len(scores)
                criterion2 = penalized_score > 0

                if criterion1 and criterion2:
                    limb = {
                        "part_a": pka,
                        "part_b": pkb,
                        "limb_score": penalized_score,
                        "limb_type": i,
                    }
                    possible_limbs.append(limb)

        # calculates valid limb connections
        num_a = len(bodyparts[partA_id])
        num_b = len(bodyparts[partB_id])
        max_connections = min(num_a, num_b)

        srt = sorted(possible_limbs, key=lambda x: x["limb_score"], reverse=True)

        cntr = 0
        used = []
        for x in srt:
            if x["part_a"]["id"] not in used and x["part_b"]["id"] not in used:
                valid_limbs.append(x)
                cntr += 1
                used.append(x["part_a"]["id"])
                used.append(x["part_b"]["id"])

                if cntr >= max_connections:
                    break
    return valid_limbs


def group_parts(valid_limbs):

    def find(my_list, item):
        for index, x in enumerate(my_list):
            if item in x:
                return index
        return None  # Item not found

    part_groups = []
    for limb in valid_limbs:
        index_a = find(part_groups, limb["part_a"]["id"])
        index_b = find(part_groups, limb["part_b"]["id"])

        if index_a is None and index_b is None:
            part_groups.append([limb["part_a"]["id"], limb["part_b"]["id"]])

        elif index_a is not None and index_b is None:
            part_groups[index_a].append(limb["part_b"]["id"])

        elif index_a is None and index_b is not None:
            part_groups[index_b].append(limb["part_a"]["id"])

        elif index_a is not None and index_b is not None:
            if index_a != index_b:
                srt = sorted((index_a, index_b), reverse=True)
                merge = part_groups[index_a] + part_groups[index_b]
                del part_groups[srt[0]]
                del part_groups[srt[1]]
                part_groups.append(merge)
    return part_groups


def group_limbs(valid_limbs, part_groups):
    limb_groups = [[] for _ in range(len(part_groups))]

    for limb in valid_limbs:
        for i in range(len(part_groups)):
            if limb["part_a"]["id"] in part_groups[i] and limb["part_b"]["id"] in part_groups[i]:
                limb_groups[i].append(limb)
                break
        else:
            # This block is executed if the 'break' statement is NOT encountered in the for loop
            print("Limb not assigned to any group:", limb)

    return limb_groups


def supress_low_conf(limb_groups):
    keep = []
    for group in limb_groups:
        score = 0
        for x in group:
            score += x["limb_score"] + x["part_a"]["score"] + x["part_b"]["score"]
        if len(group) >= 3 and (score / len(group)) > 0.2:
            keep.append(group)
    return keep


def post_process(heatmaps, pafs, image_size):
    heatmaps = transforms.functional.resize(
        heatmaps,
        tuple(reversed(image_size)),
        transforms.functional.InterpolationMode.BICUBIC,
        antialias=False,
    )

    pafs = transforms.functional.resize(
        pafs,
        tuple(reversed(image_size)),
        transforms.functional.InterpolationMode.BICUBIC,
        antialias=False,
    )

    heatmaps = heatmaps.numpy()
    pafs = pafs.numpy()

    bodyparts = get_bodyparts(heatmaps)
    valid_limbs = get_limbs(pafs, bodyparts, image_size)
    part_groups = group_parts(valid_limbs)
    limb_groups = group_limbs(valid_limbs, part_groups)
    confident = supress_low_conf(limb_groups)

    humans = []
    for group in confident:
        my_list = []
        for x in group:
            my_list.append(x["part_a"])
            my_list.append(x["part_b"])
        humans.append(my_list)

    return humans


def format(humans):
    formated = []
    for x in humans:
        kpts = [[0, 0, 0] for _ in range(18)]
        for part in x:
            kpts[part["part_type"]] = part["coords"].tolist() + [1]
        formated.append(kpts)
    return formated


def coco_format(humans):
    formated = format(humans)
    for i in range(len(formated)):
        formated[i].pop(17)  # remove neck
    coco_humans = np.array(formated).reshape(-1, 51).tolist()
    return coco_humans
