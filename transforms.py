import torch
import torchvision.transforms.v2.functional as F
from torchvision.transforms import v2
import numpy as np


class Resize():
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, larger of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, keypoints, vis = sample['image'], sample['kpt_coords'], sample['kpt_vis']
        w, h = image.size

        if isinstance(self.output_size, int):
            if h > w:
                new_w, new_h = self.output_size * w / h, self.output_size
            else:
                new_w, new_h = self.output_size, self.output_size * h / w
        else:
            new_w, new_h = self.output_size

        new_w, new_h = int(new_w), int(new_h)

        image = F.resize(image, (new_h, new_w))

        keypoints = ((keypoints + [0.5, 0.5]) * [new_w / w, new_h / h]) - [0.5, 0.5]
        return {'image': image, 'kpt_coords': keypoints, "kpt_vis": vis}


class RandomCrop():
    """Crop randomly the image in a sample.

    Args:
        crop_factor (tuple or int): Larger edge crop factor. Smaller edge stayes uncropped.
    """

    def __init__(self, crop_factor):
        self.crop_factor = crop_factor

    def __call__(self, sample):
        image, keypoints, vis = sample['image'], sample['kpt_coords'], sample['kpt_vis']
        w, h = image.size

        if h > w:
            new_w, new_h = w, self.crop_factor * h
        else:
            new_w, new_h = self.crop_factor * w, h

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        image = F.crop(image, top, left, new_h, new_w)

        keypoints = keypoints - [left, top]

        # deactivate cropped keypoints
        for i in range(len(keypoints)):
            for j in range(len(keypoints[i])):
                if (0 > keypoints[i][j][0] > new_w) or (0 > keypoints[i][j][0] > new_h):  # check > or >=
                    vis[i][j] = 0

        return {'image': image, 'kpt_coords': keypoints, "kpt_vis": vis}


class Pad():
    """Pad the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If int, square image
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, keypoints, vis = sample['image'], sample['kpt_coords'], sample['kpt_vis']
        w, h = image.size
        new_w, new_h = self.output_size

        w_pad = new_w - w
        h_pad = new_h - h

        top = 0
        bottom = h_pad
        left = 0
        right = w_pad

        image = F.pad(image, [left, top, right, bottom])

        keypoints = keypoints + [left, top]

        return {'image': image, 'kpt_coords': keypoints, "kpt_vis": vis}


def resize_keypoints(kpt_coords, stride=8):
    resized = ((kpt_coords + [0.5, 0.5]) / stride) - [0.5, 0.5]
    return resized
