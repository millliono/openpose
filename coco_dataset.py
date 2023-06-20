import os.path
from typing import Any, Callable, List, Optional, Tuple
from PIL import Image
from torchvision.datasets.vision import VisionDataset
import utils
import torch
import numpy as np

"""
    This is the coco keypoints dataset class. 
    
    keeps only images with keypoint annotations
    keeps people annotations without keypoints when other keypoint annoted people exist

    Returns:    dict of list
"""


class CocoKeypoints(VisionDataset):
    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        resize_keypoints_to=(224, 224),
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.resize_keypoints_to = resize_keypoints_to
        self.ids = self.coco.getImgIds(catIds=self.coco.getCatIds(catNms="person"))

        # retrieve images with person keypoint annotations
        self.ids = [id for id in self.ids if self.exists_keypoint_annotation(id)]

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def exists_keypoint_annotation(self, img_id):
        target = self._load_target(img_id)
        keypoint_anns = [tar for tar in target if tar["num_keypoints"] > 0]
        return True if len(keypoint_anns) > 0 else False

    def tf_resize_keypoints(self, keypoints, orig_image_size):
        # this is a transform that resizes keypoints after an image resizing transform
        target_size = self.resize_keypoints_to
        width_resize = target_size[0] / orig_image_size[0]
        height_resize = target_size[1] / orig_image_size[1]
        resized_keypoints = keypoints * np.array([width_resize, height_resize, 1])
        return resized_keypoints

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
            returns: 
                PIL image
                numpy array keypoints
        """
        id = self.ids[index]

        image = self._load_image(id)
        orig_image_size = image.size
        if self.transform is not None:
            image = self.transform(image)

        target = self._load_target(id)
        target = utils.list_of_dicts_to_dict_of_lists(target)
        num_targets = len(target["keypoints"])

        keypoints = np.array(target["keypoints"]).reshape(num_targets, 17, 3)
        keypoints = self.tf_resize_keypoints(keypoints, orig_image_size)

        return image, keypoints

    def __len__(self) -> int:
        return len(self.ids)
