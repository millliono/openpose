import torchvision.transforms.v2.functional as F
from torchvision.transforms import v2
import numpy as np
import torch


class Hflip():
    """Flip the image horizontally."""

    def __call__(self, sample):
        image, keypoints, vis = sample['image'], sample['kpt_coords'], sample['kpt_vis']
        w, h = image.size

        image = F.horizontal_flip(image)

        keypoints[:, :, 0] = w - keypoints[:, :, 0]

        new_order = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])
        keypoints = keypoints[:, new_order, :]
        vis = vis[:, new_order, :]

        return {'image': image, 'kpt_coords': keypoints, "kpt_vis": vis}


class RandomResize():
    """Randomly resize the image in a sample.

    Args:
        resize_range (tuple): resize_factor is a random number sampled from resize_range. 
            new_size = old_size * resize_factor while keeping aspect ratio.
    """

    def __init__(self, resize_range):
        self.resize_range = resize_range

    def __call__(self, sample):
        image, keypoints, vis = sample['image'], sample['kpt_coords'], sample['kpt_vis']
        w, h = image.size

        resize_factor = np.random.uniform(self.resize_range[0], self.resize_range[1])
        new_w, new_h = resize_factor * w, resize_factor * h

        new_w, new_h = int(new_w), int(new_h)

        image = F.resize(image, (new_h, new_w))

        keypoints = keypoints * [new_w / w, new_h / h]
        return {'image': image, 'kpt_coords': keypoints, "kpt_vis": vis}


class RandomCrop():
    """Crop randomly the image in a sample.

    Args:
        crop_size (int): the size of cropped output.
    """

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        image, keypoints, vis = sample['image'], sample['kpt_coords'], sample['kpt_vis']
        w, h = image.size

        top = np.random.randint(0, max(h - self.crop_size + 1, 1))
        left = np.random.randint(0, max(w - self.crop_size + 1, 1))

        image = F.crop(image, top, left, min(self.crop_size, h), min(self.crop_size, w))

        keypoints = keypoints - [left, top]

        # deactivate cropped keypoints
        for i in range(len(keypoints)):
            for j in range(len(keypoints[i])):
                if not (0 <= keypoints[i][j][0] <= self.crop_size and 0 <= keypoints[i][j][1] <= self.crop_size):
                    vis[i][j] = 0

        return {'image': image, 'kpt_coords': keypoints, "kpt_vis": vis}


class Pad():
    """Pad the image in a sample to a given size. Makes square image.

    Args:
        output_size (int): Desired output size. 
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, keypoints, vis = sample['image'], sample['kpt_coords'], sample['kpt_vis']
        w, h = image.size
        out_w = self.output_size
        out_h = self.output_size

        w_pad = out_w - w
        h_pad = out_h - h

        top = h_pad // 2
        bottom = h_pad - top
        left = w_pad // 2
        right = w_pad - left

        image = F.pad(image, [left, top, right, bottom])

        keypoints = keypoints + [left, top]

        return {'image': image, 'kpt_coords': keypoints, "kpt_vis": vis}


class RandomRotation():
    """Rotate the image in a sample by a random angle.

    Args:
        degrees (int): Range of degrees to select from.
            The range will be (-degrees, degrees).
    """

    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, sample):
        image, keypoints, vis = sample['image'], sample['kpt_coords'], sample['kpt_vis']

        theta = np.random.randint(-self.degrees, self.degrees + 1)

        image = F.rotate(image, theta)

        center = (image.width / 2, image.height / 2)
        theta = np.radians(-theta)
        rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        for i in range(len(keypoints)):
            keypoints[i] -= center
            keypoints[i] = (rot_mat @ keypoints[i].T).T
            keypoints[i] += center

        return {'image': image, 'kpt_coords': keypoints, 'kpt_vis': vis}


class ToTensor(object):
    """Convert to Tensors."""

    def __call__(self, sample):
        return {
            'image':
                v2.Compose([
                    v2.ToImage(),
                    v2.ToDtype(torch.float, scale=True),
                    # v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])(sample['image']),
            'pafs':
                torch.tensor(np.array(sample['pafs']), dtype=torch.float),
            'heatmaps':
                torch.tensor(np.array(sample['heatmaps']), dtype=torch.float)
        }
