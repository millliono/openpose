import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.ndimage.morphology import generate_binary_structure



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
    image = np.zeros((size, size))
    gaussian1 = create_gaussian_array(size, center1, std_dev1)
    image += gaussian1
    gaussian2 = create_gaussian_array(size, center2, std_dev2)
    image += gaussian2
    return image


def add_gaussian_noise(image, mean=0, std_dev=0.05):
    noise = np.random.normal(mean, std_dev, size=image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)


def get_peaks(img):
    map_left = np.zeros(img.shape)
    map_left[1:, :] = img[:-1, :]
    map_right = np.zeros(img.shape)
    map_right[:-1, :] = img[1:, :]
    map_up = np.zeros(img.shape)
    map_up[:, 1:] = img[:, :-1]
    map_down = np.zeros(img.shape)
    map_down[:, :-1] = img[:, 1:]

    peaks_binary = np.logical_and.reduce(
        (img >= map_left, img >= map_right, img >= map_up, img >= map_down, img > 0)
    )
    peaks_binary = peaks_binary.astype(int)
    return peaks_binary


image = get_image()
gauss_filtered = gaussian_filter(image, sigma=3)
gauss_peaks = get_peaks(gauss_filtered)
max_filtered = maximum_filter(image, footprint=generate_binary_structure(2, 1)) 
max_peaks = max_filtered==image


image_noise = add_gaussian_noise(image)
gauss_filtered_noise = gaussian_filter(image_noise, sigma=3)
gauss_peaks_noise = get_peaks(gauss_filtered_noise)
max_filtered_noise = maximum_filter(image_noise, footprint=generate_binary_structure(2, 1))
max_peaks_noise = (max_filtered_noise==image_noise) * (image_noise > 0.3)

fig, axs = plt.subplots(4, 3)

axs[0, 0].imshow(image)
axs[0, 0].set_title("image")

axs[0, 1].imshow(gauss_filtered)
axs[0, 1].set_title("gauss filtered")

axs[0, 2].imshow(gauss_peaks)
axs[0, 2].set_title("peaks")

# ---------------------------------
axs[1, 0].imshow(image)
axs[1, 0].set_title("image")

axs[1, 1].imshow(max_filtered)
axs[1, 1].set_title("max filtered")

axs[1, 2].imshow(max_peaks)
axs[1, 2].set_title("peaks")

# ---------------------------------------
axs[2, 0].imshow(image_noise)
axs[2, 0].set_title("noise")

axs[2, 1].imshow(gauss_filtered_noise)
axs[2, 1].set_title("gauss filtered")

axs[2, 2].imshow(gauss_peaks_noise)
axs[2, 2].set_title("peaks")

# ---------------------------------------
axs[3, 0].imshow(image_noise)
axs[3, 0].set_title("noise")

axs[3, 1].imshow(max_filtered_noise)
axs[3, 1].set_title("max filtered")

axs[3, 2].imshow(max_peaks_noise)
axs[3, 2].set_title("peaks")


plt.tight_layout()
plt.show()
