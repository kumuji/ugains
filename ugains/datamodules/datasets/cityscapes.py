from typing import Any, Callable, Dict, List, Optional, Union

import json
import random
from collections import namedtuple
from pathlib import Path

import numpy as np

from PIL import Image, ImageFile
from pycocotools import mask as mask_utils
from ugains.utils.data_utils import SharedData

# sometimes I get OSError when loading files
ImageFile.LOAD_TRUNCATED_IMAGES = True


# heavily based on https://pytorch.org/vision/stable/_modules/torchvision/datasets/cityscapes.html
# decided not to use inheritance, as it would be harder to read.
class Cityscapes:
    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple(
        "CityscapesClass",
        [
            "name",
            "id",
            "train_id",
            "category",
            "category_id",
            "has_instances",
            "ignore_in_eval",
            "color",
        ],
    )
    classes = [
        CityscapesClass("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("ego vehicle", 1, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass(
            "rectification border", 2, 255, "void", 0, False, True, (0, 0, 0)
        ),
        CityscapesClass("out of roi", 3, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("dynamic", 5, 255, "void", 0, False, True, (111, 74, 0)),
        CityscapesClass("ground", 6, 255, "void", 0, False, True, (81, 0, 81)),
        CityscapesClass("road", 7, 0, "flat", 1, False, False, (128, 64, 128)),
        CityscapesClass("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)),
        CityscapesClass("parking", 9, 255, "flat", 1, False, True, (250, 170, 160)),
        CityscapesClass("rail track", 10, 255, "flat", 1, False, True, (230, 150, 140)),
        CityscapesClass(
            "building", 11, 2, "construction", 2, False, False, (70, 70, 70)
        ),
        CityscapesClass(
            "wall", 12, 3, "construction", 2, False, False, (102, 102, 156)
        ),
        CityscapesClass(
            "fence", 13, 4, "construction", 2, False, False, (190, 153, 153)
        ),
        CityscapesClass(
            "guard rail", 14, 255, "construction", 2, False, True, (180, 165, 180)
        ),
        CityscapesClass(
            "bridge", 15, 255, "construction", 2, False, True, (150, 100, 100)
        ),
        CityscapesClass(
            "tunnel", 16, 255, "construction", 2, False, True, (150, 120, 90)
        ),
        CityscapesClass("pole", 17, 5, "object", 3, False, False, (153, 153, 153)),
        CityscapesClass(
            "polegroup", 18, 255, "object", 3, False, True, (153, 153, 153)
        ),
        CityscapesClass(
            "traffic light", 19, 6, "object", 3, False, False, (250, 170, 30)
        ),
        CityscapesClass(
            "traffic sign", 20, 7, "object", 3, False, False, (220, 220, 0)
        ),
        CityscapesClass("vegetation", 21, 8, "nature", 4, False, False, (107, 142, 35)),
        CityscapesClass("terrain", 22, 9, "nature", 4, False, False, (152, 251, 152)),
        CityscapesClass("sky", 23, 10, "sky", 5, False, False, (70, 130, 180)),
        CityscapesClass("person", 24, 11, "human", 6, True, False, (220, 20, 60)),
        CityscapesClass("rider", 25, 12, "human", 6, True, False, (255, 0, 0)),
        CityscapesClass("car", 26, 13, "vehicle", 7, True, False, (0, 0, 142)),
        CityscapesClass("truck", 27, 14, "vehicle", 7, True, False, (0, 0, 70)),
        CityscapesClass("bus", 28, 15, "vehicle", 7, True, False, (0, 60, 100)),
        CityscapesClass("caravan", 29, 255, "vehicle", 7, True, True, (0, 0, 90)),
        CityscapesClass("trailer", 30, 255, "vehicle", 7, True, True, (0, 0, 110)),
        CityscapesClass("train", 31, 16, "vehicle", 7, True, False, (0, 80, 100)),
        CityscapesClass("motorcycle", 32, 17, "vehicle", 7, True, False, (0, 0, 230)),
        CityscapesClass("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)),
        CityscapesClass(
            "license plate", -1, -1, "vehicle", 7, False, True, (0, 0, 142)
        ),
    ]
    instance_divisor = 1000
    id2trainid = np.array(
        [label.train_id for label in classes if label.train_id >= 0],
        dtype="uint8",
    )
    trainid2color = np.concatenate(
        (
            np.array(
                [
                    label.color
                    for label in classes
                    if (label.train_id >= 0 and label.train_id < 255)
                ]
            ),
            np.full((255, 3), 0),
        )
    )
    trainid2name = np.concatenate(
        (
            np.array(
                [
                    label.name
                    for label in classes
                    if (label.train_id >= 0 and label.train_id < 255)
                ]
            ),
            np.full(255, "unlabeled"),
        )
    )

    def __init__(
        self,
        root: str,
        split: str = "train",  # train, test, val
        mode: str = "fine",  # fine, coarse
        target_type: Union[List[str], str] = "instance",
        transform: Optional[Callable] = None,
        void_masks=False,
        data_fraction=1.0,
        train_with_pseudomasks=False,
    ):
        mode = "gtFine" if mode == "fine" else "gtCoarse"

        self.target_type = target_type
        self.transform = transform
        self.void_masks = void_masks
        self.split = split

        # check if we will use panoptic dataset
        self.use_mask = any(
            ["_mask" in single_target_type for single_target_type in target_type]
        )

        root = Path(root)
        images_dir = root / "leftImg8bit" / split
        if not images_dir.exists():
            raise ValueError("Images dir not found")
        targets_dir = root / mode / split
        if not targets_dir.exists():
            raise ValueError("Targets dir not found")
        images = []
        targets = []
        for image_path in sorted(images_dir.glob("*/*")):
            image_targets = []
            scene_name = image_path.stem.split("_leftImg8bit")[0]
            for target_type_request in self.target_type:
                target_suffix = self._get_target_suffix(mode, target_type_request)
                target_name = f"{scene_name}_{target_suffix}"
                image_targets.append(
                    str(targets_dir / image_path.parent.name / target_name)
                )
            images.append(str(image_path))
            targets.append(image_targets)

        self.cutoff_index = 0
        self.train_with_pseudomasks = train_with_pseudomasks
        self.data_fraction = data_fraction
        if data_fraction < 1.0:
            self.cutoff_index = int(data_fraction * len(images))
            if not train_with_pseudomasks:
                images = images[: self.cutoff_index]
                targets = targets[: self.cutoff_index]
        self.images = SharedData(images)
        self.targets = SharedData(targets)

    def _get_target_suffix(self, mode: str, target_type: str) -> str:
        if "instance" in target_type:
            return f"{mode}_instanceIds.png"
        elif "semantic" in target_type:
            return f"{mode}_labelIds.png"
        elif "color" in target_type:
            return f"{mode}_color.png"
        else:
            return f"{mode}_polygons.json"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.array(Image.open(self.images[index]).convert("RGB"))
        targets = []
        for i, t in enumerate(self.target_type):
            target = None
            if t == "polygon":
                target = self._load_json(self.targets[index][i])
            else:
                mask_target = None
                if (self.split == "train" and self.train_with_pseudomasks) and (
                    index > self.cutoff_index
                ):
                    load_path = Path(self.images[index]).stem
                    segmentation = self._load_json(
                        f"/nodes/dom/work/nekrasov/data/cityscapes_relabeled/{load_path}_rle.json"
                    )
                    mask_target = mask_utils.decode(segmentation)
                    # target = np.sum(target * np.array(range(target.shape[-1]))[None, None, :], axis=-1)
                    # target = target + 254_000
                else:
                    target = np.array(Image.open(self.targets[index][i]))
            targets.append(target)
        target = targets

        if self.transform is not None:
            if mask_target is None:
                for idx, target_type in enumerate(self.target_type):
                    if "semantic" in target_type:
                        target[idx] = self.id2trainid[target[idx]]
                target = np.stack(target).transpose(1, 2, 0)
            else:
                target = mask_target
            transformed = self.transform(image=np.array(image), mask=target)
            image = transformed["image"].transpose(2, 0, 1)
            target = transformed["mask"].transpose(2, 0, 1).astype(np.int64)

        if not self.use_mask:
            return image, target[0]

        target = self.prepare_mask_labels(target, mask_target=mask_target)
        return image, target

    def prepare_mask_labels(self, target, mask_target=None):
        return_target_dict = {"semantic": None, "labels": None, "masks": None}

        if mask_target is not None:
            if mask_target.shape[-1] > 100:
                max_masks = random.sample(
                    range(mask_target.shape[-1]), k=min(mask_target.shape[-1], 100)
                )
                mask_target = mask_target[..., max_masks]
            return_target_dict["labels"] = np.full(mask_target.shape[-1], 254)
            return_target_dict["masks"] = mask_target.transpose(2, 0, 1)
            return return_target_dict

        instance_masks_exist = any(
            [
                "instance_mask" in single_target_type
                for single_target_type in self.target_type
            ]
        )
        semantic_mask_exists = any(
            [
                "semantic_mask" in single_target_type
                for single_target_type in self.target_type
            ]
        )
        labels = []
        masks = []
        if semantic_mask_exists:
            semantic_mask_index = self.target_type.index("semantic_mask")
            semseg = target[semantic_mask_index]

        if (not instance_masks_exist) and semantic_mask_exists:
            semantic_mask_index = self.target_type.index("semantic_mask")
            for semantic_id in np.unique(target[semantic_mask_index]):
                if semantic_id == 255:
                    if self.void_masks:
                        labels.append(255)
                        masks.append(target[semantic_mask_index] == semantic_id)
                    else:
                        continue
                elif semantic_id // 254 == 1000:
                    labels.append(254)
                    masks.append(target[semantic_mask_index] == semantic_id)
                else:
                    labels.append(semantic_id)
                    masks.append(target[semantic_mask_index] == semantic_id)

        if instance_masks_exist:
            instance_mask_index = self.target_type.index("instance_mask")
            for instance_id in np.unique(target[instance_mask_index]):
                if instance_id // 254 == 1000:
                    semantic_id = 254
                elif instance_id < self.instance_divisor:
                    if instance_id > 23:
                        # it is crowd
                        continue
                    semantic_id = self.id2trainid[instance_id]
                else:
                    try:
                        semantic_id = self.id2trainid[
                            instance_id // self.instance_divisor
                        ]
                    except IndexError as e:
                        print(e)
                        print(f"instance_id: {instance_id}")
                        continue
                # if semantic_id == 255:
                #     continue
                labels.append(semantic_id)
                masks.append(target[instance_mask_index] == instance_id)

        return_target_dict["semantic"] = semseg.astype(np.uint8)
        if len(masks) == 0:
            masks = np.full((0, semseg.shape[-2], semseg.shape[-1]), 0)
        else:
            masks = np.stack(masks)
        # we put ood always as the last masks
        labels = np.array(labels)
        sorted_labels_index = np.argsort(labels)
        labels = labels[sorted_labels_index]
        masks = masks[sorted_labels_index]
        return_target_dict["labels"] = labels
        return_target_dict["masks"] = masks
        return return_target_dict

    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path) as file:
            data = json.load(file)
        return data
