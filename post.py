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
                            "id_a": pka["id"],
                            "id_b": pkb["id"],
                            "score": penalized_score,
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
                limb_scores[i], key=lambda x: x["score"], reverse=True
            )

            my_list = []
            used = []
            for x in limb_scores[i]:
                if x["id_a"] not in used and x["id_b"] not in used:
                    my_list.append(x)
                    used.append(x["id_a"])
                    used.append(x["id_b"])

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


def assign_limbs_to_people(connections):
    people = []

    for x in connections:
        if x:
            for y in x:
                index_a = find_in_list(people, y["id_a"])
                index_b = find_in_list(people, y["id_b"])

                if index_a is None and index_b is None:
                    people.append([y["id_a"], y["id_b"]])

                elif index_a is not None and index_b is None:
                    people[index_a].append(y["id_b"])

                elif index_a is None and index_b is not None:
                    people[index_b].append(y["id_a"])

                elif index_a is not None and index_b is not None:
                    if index_a == index_b:
                        continue
                    else:
                        merged = people[index_a] + people[index_b]
                        del people[index_a]
                        del people[index_b]
                        people.append(merged)
    return people


def get_people_parts(people, bodyparts):
    unpacked = []
    for x in bodyparts:
        if x:
            for y in x:
                unpacked.append(y)

    all_people_parts = []
    for x in people:
        my_list = []
        for y in x:
            my_list.append(unpacked[y])
            assert unpacked[y]["id"] == y
        all_people_parts.append(my_list)

    for i in range(len(all_people_parts)):
        all_people_parts[i] = sorted(all_people_parts[i], key=lambda x: x["part_id"])

    return all_people_parts


def body_parser(image_size, heatmaps, pafs):
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
    people = assign_limbs_to_people(connections)
    people_parts = get_people_parts(people, bodyparts)

    return people_parts


def format_keypoints(people_parts):
    formated_keypoints = []

    for x in people_parts:
        my_list = [None] * 17
        for y in x:
            my_list[y["part_id"]] = y["coords"].tolist()
        formated_keypoints.append(my_list)

    return formated_keypoints


def show_keypoints(image, formated_keypoints):
    res = show_utils.draw_keypoints(
        image, formated_keypoints, connectivity=common.connect_skeleton
    )
    plt.imshow(res)
