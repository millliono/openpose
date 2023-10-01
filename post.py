import numpy as np
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


def get_limb_scores(pafs, bodyparts):
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

                    score = np.sum(np.dot(v_norm, (paf_x, paf_y)))

                    limb = {
                        "id_a": pka["id"],
                        "id_b": pkb["id"],
                        "score": score,
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
    humans = []

    for x in connections:
        if x:
            for y in x:
                index_a = find_in_list(humans, y["id_a"])
                index_b = find_in_list(humans, y["id_b"])

                if index_a is None and index_b is None:
                    humans.append([y["id_a"], y["id_b"]])

                elif index_a is not None and index_b is None:
                    humans[index_a].append(y["id_b"])

                elif index_a is None and index_b is not None:
                    humans[index_b].append(y["id_a"])

                elif index_a is not None and index_b is not None:
                    if index_a == index_b:
                        continue
                    else:
                        merged = humans[index_a] + humans[index_b]
                        del humans[index_a]
                        del humans[index_b]
                        humans.append(merged)
    return humans


def get_people_parts(humans, bodyparts):
    unpacked = []
    for x in bodyparts:
        if x:
            for y in x:
                unpacked.append(y)

    all_people_parts = []
    for x in humans:
        my_list = []
        for y in x:
            my_list.append(unpacked[y])
            assert unpacked[y]["id"] == y
        all_people_parts.append(my_list)

    return all_people_parts
