import os.path
from typing import Any, List, Tuple
from PIL import Image
import numpy as np
from collections import defaultdict
import torch.utils.data as data
import dataset_utils
import transforms


class CocoKeypoints(data.Dataset):

    def __init__(self, root, annFile, transform):
        from pycocotools.coco import COCO

        self.root = root
        self.coco = COCO(annFile)
        self.ids = self.coco.getImgIds(catIds=self.coco.getCatIds(catNms="person"))
        self.transform = transform

        # only retrieve images that have person keypoint annotations
        self.ids = [id for id in self.ids if self.exists_keypoint_annotation(id)]

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def exists_keypoint_annotation(self, img_id):
        target = self._load_target(img_id)
        keypoint_anns = [True for x in target if x["num_keypoints"] > 0]
        return True if len(keypoint_anns) > 0 else False

    def list_of_dicts_to_dict_of_lists(self, list_of_dicts):
        dict_of_lists = defaultdict(list)
        for dct in list_of_dicts:
            for key, value in dct.items():
                dict_of_lists[key].append(value)
        return dict(dict_of_lists)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]

        image = self._load_image(id)
        target = self._load_target(id)
        anns = (image.copy(), target)

        target = [x for x in target if x["num_keypoints"] > 0]  # remove non-keypoint-annotated targets
        targ = self.list_of_dicts_to_dict_of_lists(target)

        keypoints = np.array(targ["keypoints"]).reshape(-1, 17, 3)
        coords = keypoints[:, :, :2].reshape(-1, 17, 2).astype(float)
        coords += 0.5  # floating point coordinates
        vis = keypoints[:, :, 2].reshape(-1, 17, 1)

        if self.transform:
            tf = self.transform({'image': image, 'kpt_coords': coords, 'kpt_vis': vis})
            image, coords, vis = tf['image'], tf["kpt_coords"], tf["kpt_vis"]

        coords = (coords / 8) - 0.5  # pixel coordinates
        keypoints = np.concatenate((coords, vis), axis=2).tolist()
        keypoints = dataset_utils.add_neck(keypoints, visibility=[2])

        targ_size = np.array(image.size) // 8
        heatmaps = dataset_utils.get_heatmaps(keypoints, targ_size, visibility=[1, 2])
        pafs, paf_locs = dataset_utils.get_pafs(keypoints, targ_size, visibility=[1, 2])

        ts = transforms.ToTensor()({'image': image, 'pafs': pafs, 'heatmaps': heatmaps})

        return ts['image'], ts['pafs'], ts['heatmaps'], id, paf_locs, anns

    def __len__(self) -> int:
        return len(self.ids)
