import os.path
from typing import Any, Callable, List, Optional, Tuple
from PIL import Image
import numpy as np
from collections import defaultdict
import dataset_utils
import torch.utils.data as data
import torch
import transforms
from torchvision.transforms import v2


class CocoKeypoints(data.Dataset):

    def __init__(self, root, annFile, transform, targ_size):
        from pycocotools.coco import COCO

        self.root = root
        self.coco = COCO(annFile)
        self.ids = self.coco.getImgIds(catIds=self.coco.getCatIds(catNms="person"))
        self.transform = transform
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

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]

        image = self._load_image(id)
        target = self._load_target(id)

        # mask_out = dataset_utils.get_mask_out(image.size, target, self.coco, self.targ_size)
        targ = self.list_of_dicts_to_dict_of_lists(target)

        keypoints = np.array(targ["keypoints"]).reshape(-1, 17, 3)
        kpt_coords = keypoints[:, :, :2].reshape(-1, 17, 2)
        kpt_vis = keypoints[:, :, 2].reshape(-1, 17, 1)

        tf = self.transform({'image': image, 'kpt_coords': kpt_coords, 'kpt_vis': kpt_vis})
        tf_image, tf_coords, tf_vis = tf['image'], tf["kpt_coords"], tf["kpt_vis"]

        keypoints = np.concatenate((transforms.resize_keypoints(tf_coords, stride=8), tf_vis), axis=2)
        keypoints = keypoints.tolist()
        keypoints = dataset_utils.add_neck(keypoints, visibility=[2])

        heatmaps = dataset_utils.get_heatmaps(keypoints, self.targ_size, visibility=[1, 2])
        pafs, paf_locs = dataset_utils.get_pafs(keypoints, self.targ_size, visibility=[1, 2])

        # target transforms
        tf_image = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float, scale=True)])(tf_image)
        heatmaps = torch.tensor(np.array(heatmaps), dtype=torch.float)
        pafs = torch.tensor(np.array(pafs), dtype=torch.float)

        return tf_image, pafs, heatmaps, image, paf_locs, target,  #mask_out

    def __len__(self) -> int:
        return len(self.ids)
