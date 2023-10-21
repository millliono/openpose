import os.path
from typing import Any, Callable, List, Optional, Tuple
from PIL import Image
import numpy as np
from collections import defaultdict
import dataset_utils
import torch.utils.data as data
import torch


class CocoKeypoints(data.Dataset):

    def __init__(self, root, annFile, targ_size=(46, 46), input_transform=None, train=True):
        from pycocotools.coco import COCO

        self.root = root
        self.coco = COCO(annFile)
        self.ids = self.coco.getImgIds(catIds=self.coco.getCatIds(catNms="person"))
        self.input_transform = input_transform
        self.train = train
        self.targ_size = targ_size

        # only retrieve images that have person keypoint annotations
        self.ids = [id for id in self.ids if self.exists_keypoint_annotation(id)]

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def exists_keypoint_annotation(self, img_id):
        target = self._load_target(img_id)
        keypoint_anns = [x for x in target if x["num_keypoints"] > 0]
        return True if len(keypoint_anns) > 0 else False

    def list_of_dicts_to_dict_of_lists(self, list_of_dicts):
        dict_of_lists = defaultdict(list)
        for dct in list_of_dicts:
            for key, value in dct.items():
                dict_of_lists[key].append(value)
        return dict(dict_of_lists)

    def resize_keypoints(self, keypoints, old_size, new_size):
        scale_x = old_size[0] / new_size[0]
        scale_y = old_size[1] / new_size[1]

        visibility = keypoints[:, :, 2].reshape(-1, 17, 1)
        visible = np.where(visibility > 0, 1, 0)

        coords = keypoints[:, :, :2].reshape(-1, 17, 2)
        resized = (coords + np.array([0.5, 0.5])) / np.array([scale_x, scale_y]) - np.array([0.5, 0.5])

        resized = resized * visible
        res = np.concatenate((resized, visibility), axis=2)
        return res

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]

        image = self._load_image(id)
        target = self._load_target(id)

        # input transforms
        tf_image = self.input_transform(image)

        if not self.train:
            return tf_image, torch.tensor(image.size), id

        mask_out = dataset_utils.get_mask_out(image, target, self.coco, self.targ_size)

        targ = self.list_of_dicts_to_dict_of_lists(target)

        keypoints = np.array(targ["keypoints"]).reshape(-1, 17, 3)
        keypoints = self.resize_keypoints(keypoints, image.size, self.targ_size)
        keypoints = keypoints.tolist()

        heatmaps = dataset_utils.get_heatmaps(keypoints, self.targ_size, visibility=[1, 2])
        pafs = dataset_utils.get_pafs(keypoints, self.targ_size, visibility=[1, 2])

        # target transforms
        heatmaps = torch.tensor(np.array(heatmaps), dtype=torch.float32)
        pafs = torch.tensor(np.array(pafs), dtype=torch.float32)

        return tf_image, pafs, heatmaps, mask_out, keypoints, image, target

    def __len__(self) -> int:
        return len(self.ids)
