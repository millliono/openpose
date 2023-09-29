import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.ndimage import generate_binary_structure
import common


def get_bodyparts(heatmaps):
    all_bodyparts = []
    unique_id = 0

    for i in range(len(heatmaps)):
        filtered = maximum_filter(
            heatmaps[i], footprint=generate_binary_structure(2, 1)
        )
        peaks_coords = np.nonzero((filtered == heatmaps[i]) * (heatmaps[i] > 0.3))

        # if no peaks found
        if peaks_coords[0].size == 0:
            all_bodyparts.append([])

        else:
            bodyparts_form_heatmap = []
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
                bodyparts_form_heatmap.append(bodypart)
            all_bodyparts.append(bodyparts_form_heatmap)

    return all_bodyparts


def get_limb_scores(pafs, parts):
    all_limb_scores = []

    for i in range(len(pafs) // 2):
        partA_id = common.connect_skeleton[i][0]
        partB_id = common.connect_skeleton[i][1]

        partsA = parts[partA_id]  # list with bodypart dictionaries
        partsB = parts[partB_id]

        pafx = pafs[2 * i]
        pafy = pafs[2 * i + 1]

        if partsA and partsB:
            limb_scores = []
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

                    # flip for ij coords
                    paf_x = pafx[line_y, line_x]
                    paf_y = pafy[line_y, line_x]

                    score = np.sum(np.dot(v_norm, (paf_x, paf_y)))

                    limb = {
                        "id_a": pka["id"],
                        "id_b": pkb["id"],
                        "score": score,
                        "limb_id": i,
                    }

                    limb_scores.append(limb)
            all_limb_scores.append(limb_scores)
        else:
            all_limb_scores.append([])

    return all_limb_scores


def get_connections(limb_scores, parts):
    all_connections = []

    for i in range(len(limb_scores)):
        partA_id = common.connect_skeleton[i][0]
        partB_id = common.connect_skeleton[i][1]

        npks_a = len(parts[partA_id])
        npks_b = len(parts[partB_id])

        if npks_a == 0 or npks_b == 0:
            all_connections.append([])
        else:
            candidate_limbs = limb_scores[i]

            max_connections = min(npks_a, npks_b)

            connections = []
            used = []
            candidate_limbs = sorted(
                candidate_limbs, key=lambda x: x["score"], reverse=True
            )

            for limb in candidate_limbs:
                if limb["id_a"] not in used and limb["id_b"] not in used:
                    connections.append(limb)
                    used.append(limb["id_a"])
                    used.append(limb["id_b"])

                    if len(connections) >= max_connections:
                        break
            all_connections.append(connections)

    return all_connections


def find_item(my_list, item):
    for index, sublist in enumerate(my_list):
        if item in sublist:
            return index
    return None  # Item not found


def assign_limbs_to_people(connections):
    humans = []

    for con in connections:
        if con:
            for limb in con:
                index_a = find_item(humans, limb["id_a"])
                index_b = find_item(humans, limb["id_b"])

                if index_a is None and index_b is None:
                    humans.append([limb["id_a"], limb["id_b"]])

                elif index_a is not None and index_b is None:
                    humans[index_a].append(limb["id_b"])

                elif index_a is None and index_b is not None:
                    humans[index_b].append(limb["id_a"])

                elif index_a is not None and index_b is not None:
                    if index_a == index_b:
                        continue
                    else:
                        merged = humans[index_a] + humans[index_b]
                        del humans[index_a]
                        del humans[index_b]
                        humans.append(merged)
    return humans