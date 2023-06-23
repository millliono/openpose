import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont


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


def show2(target):
    """
    shows all (17) heatmap or paf channels using subplots
    """
    num_images = len(target)
    num_cols = 4
    num_rows = (num_images - 1) // num_cols + 1

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 9))

    if num_rows == 1:
        axes = axes.reshape(1, -1)
    elif num_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, image in enumerate(target):
        row_idx = i // num_cols
        col_idx = i % num_cols
        axes[row_idx, col_idx].imshow(image)
        axes[row_idx, col_idx].axis("off")

    plt.tight_layout()


def show3(image1, heatmaps, pafs):
    """
    shows image with all heatmaps and all pafs stacked
    """
    plt.imshow(image1)

    alpha = 0.2
    image2 = np.add.reduce(heatmaps)
    plt.imshow(image2, cmap="gray", alpha=alpha)

    image3 = np.add.reduce(pafs)
    plt.imshow(image3, cmap="viridis", alpha=alpha)

    plt.show()


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


coco_keypoints = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
connect_skeleton = [
    (0, 1),  # nose -> left_eye
    (0, 2),  # nose -> right_eye
    (1, 3),  # left_eye -> left_ear
    (2, 4),  #  right_eye -> right_ear
    (0, 5),  # nose -> left_shoulder
    (0, 6),  # nose -> right_shoulder
    (5, 7),  # left_shoulder -> left_elbow
    (6, 8),  # right_shoulder -> right_elbow
    (7, 9),  # left_elbow -> left_wrist
    (8, 10),  # right_elbow -> right_wrist
    (5, 11),  # left_shoulder -> left_hip
    (6, 12),  # right_shoulder -> right_hip
    (11, 13),  # left_hip -> left_knee
    (12, 14),  # right_hip -> right_knee
    (13, 15),  # left_knee -> left_ankle
    (14, 16),  # right_knee -> right_ankle
]
