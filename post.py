import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.ndimage import generate_binary_structure



def surf_heatmap(heatmap):
    # Create meshgrid for X and Y dimensions
    x_dim = np.arange(0, heatmap.shape[1], 1)
    y_dim = np.arange(0, heatmap.shape[0], 1)
    X, Y = np.meshgrid(x_dim, y_dim)

    # Flatten the 2D array into a 1D array for the Z dimension
    Z = heatmap.flatten()

    # Create a figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the 3D surface
    ax.plot_surface(X, Y, Z.reshape(heatmap.shape), cmap="viridis")

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("surf plot")

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

