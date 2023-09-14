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
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
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

    def tf_resize_keypoints(self, keypoints, prev_size, new_size):
        # this is a transform that resizes keypoints after an image resizing transform
        scale_x = new_size[0] / prev_size[0]
        scale_y = new_size[1] / prev_size[1]
        resized_keypoints = keypoints * np.array([scale_x, scale_y, 1])
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
        prev_size = image.size
        if self.transform is not None:
            image = self.transform(image)
        new_size = image.size
        targ_size = (new_size[0] // 8, new_size[1] // 8)

        target = self._load_target(id)
        target = self.list_of_dicts_to_dict_of_lists(target)

        keypoints = np.array(target["keypoints"]).reshape(-1, 17, 3)
        keypoints = self.tf_resize_keypoints(keypoints, prev_size, targ_size)
        keypoints = keypoints.tolist()

        #
        # HERE paf & heatmaps is list of numpy arrays
        #
        heatmaps = dataset_utils.get_heatmaps(
            keypoints, size=targ_size, visibility=[1, 2]
        )
        pafs = dataset_utils.get_pafs(keypoints, size=targ_size, visibility=[1, 2])

        #
        # targets converted to TENSOR
        #
        image = fcn.pil_to_tensor(image)
        image = fcn.convert_image_dtype(image, torch.float32)
        pafs = torch.tensor(np.array(pafs), dtype=torch.float32)
        heatmaps = torch.tensor(np.array(heatmaps), dtype=torch.float32)
        keypoints = torch.tensor(keypoints, dtype=torch.float32)

        return image, pafs, heatmaps, keypoints, targ_size

    def __len__(self) -> int:
        return len(self.ids)
