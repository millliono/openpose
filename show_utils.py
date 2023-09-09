import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont
from torchvision import transforms
import common


def show1(imgs):
    """
    shows a list of tensor images using subplots
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def show_subplots(img):
    num_images = len(img)
    num_cols = 4
    num_rows = (num_images - 1) // num_cols + 1

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 9))

    if num_rows == 1:
        axes = axes.reshape(1, -1)
    elif num_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, htmp in enumerate(img):
        row_idx = i // num_cols
        col_idx = i % num_cols
        axes[row_idx, col_idx].imshow(htmp)

    plt.tight_layout()


def show_heatmaps(heatmaps):
    """
    shows all (17) heatmap points using subplots
    """
    show_subplots(heatmaps)


def show_heatmaps_combined(heatmaps):
    """
    shows image with all heatmaps stacked
    """
    htmp = torch.sum(heatmaps, dim=0)
    plt.imshow(htmp)


def show_pafs(pafs):
    """
    shows all (32) pafs in single figure (x_ccord, y_coord)
    using subplots
    """
    show_subplots(pafs)


def show_pafs_combined(pafs):
    """
    shows image with all paf vectors positions stacked.
    """
    paf_x = []
    for i in range(0, pafs.size(dim=0), 2):
        paf_x.append(pafs[i])

    paf_x = torch.stack(paf_x)
    paf_pos = torch.sum(paf_x, dim=0)
    paf_pos = np.where(paf_pos != 0, 1, 0)
    plt.imshow(paf_pos)


def show_pafs_quiver(pafs):
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
        px, py = np.meshgrid(np.arange(28), np.arange(28))
        row_idx = i // num_cols
        col_idx = i % num_cols
        axes[row_idx, col_idx].quiver(px, py, paf_x[i], paf_y[i], scale=1, scale_units='xy', angles='xy', pivot='tail')
        axes[row_idx, col_idx].invert_yaxis()

    plt.tight_layout()


def show_pafs_quiver_combined(pafs):
    paf_x = []
    for i in range(0, pafs.size(dim=0), 2):
        paf_x.append(pafs[i])

    paf_y = []
    for i in range(1, pafs.size(dim=0), 2):
        paf_y.append(pafs[i])

    paf_x = torch.sum(torch.stack(paf_x), dim=0)
    paf_y = torch.sum(torch.stack(paf_y), dim=0)

    px, py = np.meshgrid(np.arange(28), np.arange(28))
    plt.quiver(px, py, paf_x, paf_y, scale=1, scale_units='xy', angles='xy', pivot='tail')
    plt.gca().invert_yaxis()


def show_annotated(image, keypoints):
    res = draw_keypoints(
        F.convert_image_dtype(image, torch.uint8),
        keypoints,
        visibility=[1, 2],
        connectivity=common.connect_skeleton,
    )
    show1(res)


@torch.no_grad()
def draw_keypoints(
    image,
    keypoints,
    visibility,
    connectivity,
    keypoint_color="blue",
    line_color="yellow",
    radius: int = 2,
    width: int = 2,
):
    """
    Draws Keypoints on given RGB image.
    The values of the input image should be uint8 between 0 and 255.

    Args:
        image (Tensor): Tensor of shape (3, H, W) and dtype uint8.
        keypoints (Tensor): Tensor of shape (num_instances, K, 2) the K keypoints location for each of the N instances,

    Returns:
        img (Tensor[C, H, W]): Image Tensor of dtype uint8 with keypoints drawn.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"The image must be a tensor, got {type(image)}")
    elif image.dtype != torch.uint8:
        raise ValueError(f"The image dtype must be uint8, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size()[0] != 3:
        raise ValueError("Pass an RGB image. Other Image formats are not supported")

    if keypoints.ndim != 3:
        raise ValueError("keypoints must be of shape (num_instances, K, 2)")

    ndarr = image.permute(1, 2, 0).cpu().numpy()
    img_to_draw = Image.fromarray(ndarr)
    draw = ImageDraw.Draw(img_to_draw)
    img_kpts = keypoints.to(torch.int64).tolist()

    for kpt_id, kpt_inst in enumerate(img_kpts):
        for inst_id, kpt in enumerate(kpt_inst):
            if kpt[2] in visibility:  # change here
                x1 = kpt[0] - radius
                x2 = kpt[0] + radius
                y1 = kpt[1] - radius
                y2 = kpt[1] + radius
                draw.ellipse(
                    [x1, y1, x2, y2], fill=keypoint_color, outline=None, width=0
                )

        if connectivity:
            for connection in connectivity:
                if (
                    kpt_inst[connection[0]][2] in visibility
                    and kpt_inst[connection[1]][2] in visibility
                ):
                    start_pt_x = kpt_inst[connection[0]][0]
                    start_pt_y = kpt_inst[connection[0]][1]

                    end_pt_x = kpt_inst[connection[1]][0]
                    end_pt_y = kpt_inst[connection[1]][1]

                    draw.line(
                        ((start_pt_x, start_pt_y), (end_pt_x, end_pt_y)),
                        width=width,
                        fill=line_color,
                    )

    return (
        torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)
    )
