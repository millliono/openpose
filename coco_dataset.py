import os.path
from typing import Any, Callable, List, Optional, Tuple
from PIL import Image
from torchvision.datasets.vision import VisionDataset
import numpy as np
from collections import defaultdict
import dataset_utils
import torchvision.transforms.functional as fcn
import torch

"""
    This is the coco keypoints dataset class. 
    
    keeps only images with keypoint annotations
    keeps people annotations without keypoints when other keypoint annoted people exist

    Returns:    image: pil iamge
                keypoints: list with keypoints
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

        # only retrieve images that have person keypoint annotations
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

    def list_of_dicts_to_dict_of_lists(self, list_of_dicts):
        dict_of_lists = defaultdict(list)
        for dct in list_of_dicts:
            for key, value in dct.items():
                dict_of_lists[key].append(value)
        return dict(dict_of_lists)

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
        target = self.list_of_dicts_to_dict_of_lists(target)

        keypoints = np.array(target["keypoints"]).reshape(-1, 17, 3)
        keypoints = self.tf_resize_keypoints(keypoints, orig_image_size)
        keypoints = keypoints.tolist()

        #
        # HERE paf & heatmaps is list of numpy arrays, NOT TENSORS
        #
        pafs = dataset_utils.get_pafs(keypoints)
        heatmaps = dataset_utils.get_heatmaps(keypoints)

        #
        # example converted to TENSOR
        #
        image = fcn.pil_to_tensor(image)
        pafs = torch.tensor(np.array(pafs))
        heatmaps = torch.tensor(np.array(heatmaps))


        return image, pafs, heatmaps, keypoints

    def __len__(self) -> int:
        return len(self.ids)
