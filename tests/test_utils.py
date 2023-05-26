import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.ndimage import generate_binary_structure
from mpl_toolkits.mplot3d import Axes3D


def get_image():
    def create_gaussian_array(size, center, std_dev):
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        gaussian = np.exp(-0.5 * (dist / std_dev) ** 2)
        return gaussian

    size = 28
    center1 = (5, 25)
    std_dev1 = 1.0
    center2 = (14, 14)
    std_dev2 = 2.0
    center3 = (15, 25)
    std_dev3 = 1.0
    image = np.zeros((size, size))
    gaussian1 = create_gaussian_array(size, center1, std_dev1)
    image += gaussian1
    gaussian2 = create_gaussian_array(size, center2, std_dev2)
    image += gaussian2
    gaussian3 = create_gaussian_array(size, center3, std_dev3)
    image += gaussian3
    return image


def add_gaussian_noise(image, mean=0, std_dev=0.05):
    noise = np.random.normal(mean, std_dev, size=image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)


def plot_2d_array_in_3d(array_2d):
    # Create meshgrid for X and Y dimensions
    x_dim = np.arange(0, array_2d.shape[1], 1)
    y_dim = np.arange(0, array_2d.shape[0], 1)
    X, Y = np.meshgrid(x_dim, y_dim)

    # Flatten the 2D array into a 1D array for the Z dimension
    Z = array_2d.flatten()

    # Create a figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the 3D surface
    ax.plot_surface(X, Y, Z.reshape(array_2d.shape), cmap="viridis")

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("2D Array plotted in 3D")

    # Show the plot
    plt.show()


def get_peaks_from_heatmap(heatmap):
    filtered = maximum_filter(heatmap, footprint=generate_binary_structure(2, 1))
    peaks_of_heatmap = (filtered == heatmap) * (heatmap > 0.3)
    peaks_coords = np.nonzero(peaks_of_heatmap)
    peaks_scores = heatmap[peaks_coords]
    num_peaks = len(peaks_coords[0])
    return (np.array(peaks_coords).T, peaks_scores, num_peaks)


def get_associations(paf, peaks_a, peaks_b):
    # NOTE: implementation uses upsampled paf
    associations = []
    for i, pka in enumerate(peaks_a):
        for j, pkb in enumerate(peaks_b):
            # normal vector
            v = pkb - pka
            v_magn = np.sqrt(v[0] ** 2 + v[1] ** 2) + 1e-8
            v_norm = v / v_magn

            # line
            line_x = np.round(np.linspace(pka[0], pkb[0], 10)).astype(int)
            line_y = np.round(np.linspace(pka[1], pkb[1], 10)).astype(int)

            paf_x = paf[0][line_x, line_y]
            paf_y = paf[1][line_x, line_y]

            score = np.sum(np.dot(v_norm, (paf_x, paf_y)))
            associations.append([i, j, score])
    return associations


def get_connections(paf, peaks_a, peaks_b):
    npks_a = len(peaks_a)
    npks_b = len(peaks_b)

    if npks_a == 0 or npks_b == 0:
        return []
    else:
        associations = get_associations(paf, peaks_a, peaks_b)

        max_connections = min(npks_a, npks_b)

        connections = []
        used_pka = []
        used_pkb = []
        associations = sorted(associations, key=lambda x: x[2], reverse=True)

        for i in associations:
            if i[0] not in used_pka and i[1] not in used_pkb:
                connections.append(i)
                used_pka.append(i[0])
                used_pkb.append(i[1])

                if len(connections) >= max_connections:
                    break
    return connections

class coco_part():
    nose = 0
    neck = 1
    r_shoulder = 2
    r_elbow = 3
    r_wrist = 4
    l_shoulder = 5
    l_elbow = 6
    l_wrist = 7
    r_hip = 8
    r_knee = 9
    r_ankle = 10
    l_hip = 11
    l_knee = 12
    l_ankle = 13
    r_eye = 14
    l_eye = 15
    r_ear = 16
    l_ear = 17
    background = 18

