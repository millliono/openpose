import test_utils as f


image = f.get_image()
f.plot_2d_array_in_3d(image)

(coords, scores, num) = f.get_peaks_from_heatmap(image)
print(coords, scores, num)

# image_noise = add_gaussian_noise(image)
# max_filtered_noise = maximum_filter(image_noise, footprint=generate_binary_structure(2, 1))
# max_peaks_noise = (max_filtered_noise==image_noise) * (image_noise > 0.3)


# fig, axs = plt.subplots(2, 3)
# # ---------------------------------
# axs[0, 0].imshow(image)
# axs[0, 0].set_title("image")

# axs[0, 1].imshow(max_filtered)
# axs[0, 1].set_title("max filtered")

# axs[0, 2].imshow(max_peaks)
# axs[0, 2].set_title("peaks")

# # ---------------------------------------
# axs[1, 0].imshow(image_noise)
# axs[1, 0].set_title("noise")

# axs[1, 1].imshow(max_filtered_noise)
# axs[1, 1].set_title("max filtered")

# axs[1, 2].imshow(max_peaks_noise)
# axs[1, 2].set_title("peaks")


# plt.tight_layout()
# plt.show()
