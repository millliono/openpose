import torch
import torchvision.transforms.v2.functional as F
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

    def __call__(self, image, keypoints):
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
        return image, keypoints


class RandomCrop():
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image, keypoints):
        w, h = image.size
        new_w, new_h = self.output_size

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        image = F.crop(image, top, left, new_h, new_w)

        keypoints = keypoints - [left, top]

        # !!!!what happens to keypoints outside crop!!!!

        return image, keypoints


class Pad():
    """Pad the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image, keypoints):
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

        return image, keypoints


class ToTensor():

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'landmarks': torch.from_numpy(landmarks)}
