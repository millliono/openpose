import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import numpy as np
from PIL import ImageDraw


def show_coco(image, target, coco, draw_bbox):
    plt.axis("off")
    plt.imshow(np.asarray(image))
    # Plot segmentation and bounding box.
    coco.showAnns(target, draw_bbox)


def show_tensors(imgs):
    """
    shows a list of tensor images using subplots
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    if len(imgs) > 1:
        fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(30, 9))
    else:
        fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(4, 4))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def show_heatmaps(heatmaps):
    """
    shows all (17) heatmap points using subplots
    """
    my_list = []
    for x in heatmaps:
        my_list.append(x)

    show_tensors(my_list)


def show_heatmaps_combined(heatmaps):
    """
    shows image with all heatmaps stacked
    """
    htmp = torch.sum(heatmaps, dim=0)
    show_tensors(htmp)


def show_pafs(pafs):
    """
    shows all (32) pafs in single figure (x_ccord, y_coord)
    using subplots
    """
    my_list = []
    for x in pafs:
        my_list.append(x)
    show_tensors(my_list)


def show_pafs_quiver(pafs, size):
    """
    shows all (16) pafs as vector fields using subplots
    """
    num_images = pafs.size(dim=0) / 2
    num_cols = 4
    num_rows = (num_images - 1) // num_cols + 1

    fig, axes = plt.subplots(int(num_rows), num_cols, figsize=(12, 9))

    if num_rows == 1:
        axes = axes.reshape(1, -1)
    elif num_cols == 1:
        axes = axes.reshape(-1, 1)

    paf_x = []
    for i in range(0, pafs.size(dim=0), 2):
        paf_x.append(pafs[i])

    paf_y = []
    for i in range(1, pafs.size(dim=0), 2):
        paf_y.append(pafs[i])

    for i in range(len(paf_x)):
        px, py = np.meshgrid(np.arange(size[1]), np.arange(size[0]))
        row_idx = i // num_cols
        col_idx = i % num_cols
        axes[row_idx, col_idx].quiver(
            px,
            py,
            paf_x[i],
            paf_y[i],
            scale=1,
            scale_units="xy",
            angles="xy",
            pivot="tail",
        )
        axes[row_idx, col_idx].invert_yaxis()

    plt.tight_layout()


def show_pafs_quiver_combined(pafs, size):
    paf_x = []
    for i in range(0, pafs.size(dim=0), 2):
        paf_x.append(pafs[i])

    paf_y = []
    for i in range(1, pafs.size(dim=0), 2):
        paf_y.append(pafs[i])

    paf_x = torch.sum(torch.stack(paf_x), dim=0)
    paf_y = torch.sum(torch.stack(paf_y), dim=0)

    px, py = np.meshgrid(np.arange(size[1]), np.arange(size[0]))
    plt.quiver(
        px, py, paf_x, paf_y, scale=1, scale_units="xy", angles="xy", pivot="tail"
    )
    plt.gca().invert_yaxis()


@torch.no_grad()
def draw_keypoints(
    image,
    keypoints,
    connectivity,
    keypoint_color="blue",
    line_color="yellow",
    radius: int = 5,
    width: int = 5,
):
    img_to_draw = image
    draw = ImageDraw.Draw(img_to_draw)
    img_kpts = keypoints

    for kpt_id, kpt_inst in enumerate(img_kpts):
        for inst_id, kpt in enumerate(kpt_inst):
            if kpt is not None:
                x1 = kpt[0] - radius
                x2 = kpt[0] + radius
                y1 = kpt[1] - radius
                y2 = kpt[1] + radius
                draw.ellipse([x1, y1, x2, y2], fill=keypoint_color, outline=None, width=0)

        if connectivity:
            for connection in connectivity:
                    if kpt_inst[connection[0]] is not None and kpt_inst[connection[1]] is not None:
                        start_pt_x = kpt_inst[connection[0]][0]
                        start_pt_y = kpt_inst[connection[0]][1]

                        end_pt_x = kpt_inst[connection[1]][0]
                        end_pt_y = kpt_inst[connection[1]][1]

                        draw.line(
                            ((start_pt_x, start_pt_y), (end_pt_x, end_pt_y)),
                            width=width,
                            fill=line_color,
                        )

    return np.array(img_to_draw)


def surf_heatmap(heatmap):
    x_dim = np.arange(0, heatmap.shape[1], 1)
    y_dim = np.arange(0, heatmap.shape[0], 1)
    X, Y = np.meshgrid(x_dim, y_dim)
    Z = heatmap.flatten()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, Z.reshape(heatmap.shape), cmap="viridis")

    ax.set_title("surf plot")
