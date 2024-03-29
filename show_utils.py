import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageDraw, Image, ImageOps
import random


def show_coco(image, target, coco, draw_bbox):
    plt.axis("off")
    plt.imshow(np.asarray(image))
    coco.showAnns(target, draw_bbox)


def blend(conf_maps, image, rows, cols, figsize):
    conf_maps = [Image.fromarray((x * 255)).convert("L") for x in conf_maps]
    conf_maps = [ImageOps.colorize(x, black="blue", white="orange") for x in conf_maps]
    conf_maps = [x.resize(image.size, resample=Image.NEAREST) for x in conf_maps]
    blended = [Image.blend(x, image, 0.5) for x in conf_maps]
    plot_grid(blended, rows, cols, figsize)


def plot_grid(images, rows, cols, figsize):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i])
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()


def surface(image):
    x_dim = np.arange(0, image.shape[1], 1)
    y_dim = np.arange(0, image.shape[0], 1)
    X, Y = np.meshgrid(x_dim, y_dim)
    Z = image.flatten()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, Z.reshape(image.shape), cmap="viridis")


@torch.no_grad()
def draw_skeleton(
    image,
    coco_humans,
    connectivity,
    radius: int = 1.5,
    width: int = 2,
):
    img_to_draw = image
    draw = ImageDraw.Draw(img_to_draw)

    for x in coco_humans:
        # draw limbs
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        for connection in connectivity:
            if x[connection[0]][2] != 0 and x[connection[1]][2] != 0:
                start_pt_x = x[connection[0]][0]
                start_pt_y = x[connection[0]][1]

                end_pt_x = x[connection[1]][0]
                end_pt_y = x[connection[1]][1]

                draw.line(
                    ((start_pt_x, start_pt_y), (end_pt_x, end_pt_y)),
                    width=width,
                    fill=color,
                )
        # draw keypoints
        for y in x:
            if y[2] != 0:
                x1 = y[0] - radius
                x2 = y[0] + radius
                y1 = y[1] - radius
                y2 = y[1] + radius
                draw.ellipse(
                    [x1, y1, x2, y2],
                    outline=None,
                    width=0,
                    fill="orange",
                )

    img = np.array(img_to_draw)
    plt.imshow(img)


def pafs_quiver(pafs, size):
    num_images = len(pafs) // 2
    num_cols = 4
    num_rows = (num_images - 1) // num_cols + 1

    fig, axes = plt.subplots(int(num_rows), num_cols, figsize=(18, 18))

    if num_rows == 1:
        axes = axes.reshape(1, -1)
    elif num_cols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(len(pafs) // 2):
        pafx = pafs[2 * i]
        pafy = pafs[2 * i + 1]

        px, py = np.meshgrid(np.arange(size[1]), np.arange(size[0]))
        row_idx = i // num_cols
        col_idx = i % num_cols
        axes[row_idx, col_idx].quiver(
            px,
            py,
            pafx,
            pafy,
            scale=1,
            scale_units="xy",
            angles="xy",
            pivot="tail",
        )
        axes[row_idx, col_idx].invert_yaxis()
    plt.tight_layout()


def pafs_quiver_combined(pafs, size):
    paf_x = pafs[[x for x in range(len(pafs)) if x % 2 == 0]]
    paf_y = pafs[[x for x in range(len(pafs)) if x % 2 == 1]]

    paf_x = np.sum(paf_x, axis=0)
    paf_y = np.sum(paf_y, axis=0)

    px, py = np.meshgrid(np.arange(size[1]), np.arange(size[0]))
    plt.quiver(px, py, paf_x, paf_y, scale=1, scale_units="xy", angles="xy", pivot="tail")
    plt.gca().invert_yaxis()
