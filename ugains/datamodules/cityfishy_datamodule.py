from typing import List, Optional, Union

import numpy as np

import albumentations as A
import torch
from ugains.datamodules.datasets.cityscapes import Cityscapes
from ugains.datamodules.datasets.fishyscapes_lost_found import (
    FishyscapesLostFoundDataset,
    FishyscapesStaticDataset,
)
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class CityFishyDataModule(LightningDataModule):
    def __init__(
        self,
        fishyscapes_data_root: str = "./data/fs_lost_found",
        cityscapes_data_root: str = "./data/cityscapes",
        mode: str = "lost_and_found",  # ["lost_and_found", "static"]
        fishy_mode: str = "*",  # * for fs laf, test for laf
        target_type: Union[
            List[str], str
        ] = "semantic",  # ["semantic", "instance", "semantic_masks", "semantic_instance"]
        train_batch_size: int = 8,
        train_num_workers: int = 0,
        train_pin_memory: bool = False,
        train_image_size=(512, 1024),
        validation_batch_size: int = 8,
        validation_num_workers: int = 0,
        validation_pin_memory: bool = False,
        validation_image_size=(512, 1024),
        num_classes=19,
        ignore_label=255,
        mean_std=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        void_masks=False,
        train_data_fraction=1.0,
        train_with_pseudomasks=False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.target_type = target_type
        self.fishyscapes_data_root = fishyscapes_data_root
        self.cityscapes_data_root = cityscapes_data_root

        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.train_pin_memory = train_pin_memory
        self.validation_batch_size = validation_batch_size
        self.validation_num_workers = validation_num_workers
        self.validation_pin_memory = validation_pin_memory
        self.train_data_fraction = train_data_fraction
        self.train_with_pseudomasks = train_with_pseudomasks

        self.ignore_label = ignore_label
        self.mean_std = mean_std

        self.val_city_dataset = Cityscapes
        self.train_city_dataset = Cityscapes
        self.fishy_dataset = FishyscapesLostFoundDataset
        if mode == "static":
            self.fishy_dataset = FishyscapesStaticDataset
        self.fishy_mode = fishy_mode

        self.void_masks = void_masks

        # detectron2 augmentaitons
        max_sizes = [int(x * 0.1 * 1024) for x in range(5, 21)]
        self.transforms = A.Compose(
            [
                A.OneOf(
                    [A.SmallestMaxSize(max_size=max_size) for max_size in max_sizes],
                    p=1,
                ),
                A.RandomCrop(*train_image_size, always_apply=True),
                A.ColorJitter(
                    brightness=32 / 255,
                    contrast=0.5,
                    saturation=0.5,
                    hue=18 / 255,
                    p=0.5,
                ),  # from detectron2 ColorAugSSDTransofmm
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=mean_std[0], std=mean_std[1]),
            ],
        )
        self.target_transforms = A.Compose(
            [
                A.Resize(*validation_image_size),
                A.Normalize(mean=mean_std[0], std=mean_std[1]),
            ]
        )

    @property
    def inlier_id2trainid(self):
        return self.data_train.id2trainid

    @property
    def inlier_trainid2color(self):
        return self.data_train.trainid2color

    @property
    def inlier_trainid2name(self):
        return self.data_train.trainid2name

    @property
    def ood_id2trainid(self):
        return self.data_test.id2trainid

    @property
    def ood_trainid2color(self):
        return self.data_test.trainid2color

    @property
    def ood_trainid2name(self):
        return self.data_test.trainid2name

    @staticmethod
    def collate_masks_or_list_of_masks(batch):
        images = []
        targets = []
        for im, target in batch:
            images.append(im)
            if isinstance(target, np.ndarray):
                targets = torch.from_numpy(target).long()
            else:
                for k, v in target.items():
                    if not isinstance(target[k], np.ndarray):
                        continue
                    target[k] = torch.from_numpy(v).long()
                targets.append(target)
        images = np.stack(images)
        images = torch.from_numpy(images).float()
        return images, targets

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        # self.data_train = self.train_city_dataset(
        #     root=self.cityscapes_data_root,
        #     split="train",
        #     mode="fine",
        #     transform=self.transforms,
        #     target_type=self.target_type,
        #     void_masks=self.void_masks,
        #     data_fraction=self.train_data_fraction,
        #     train_with_pseudomasks=self.train_with_pseudomasks,
        # )
        # self.data_val = self.val_city_dataset(
        #     root=self.cityscapes_data_root,
        #     split="val",
        #     mode="fine",
        #     transform=self.target_transforms,
        #     target_type=self.target_type,
        # )
        self.data_test = self.fishy_dataset(
            root=self.fishyscapes_data_root,
            transform=self.target_transforms,
            target_type=self.target_type,
            mode=self.fishy_mode,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.train_batch_size,
            num_workers=self.train_num_workers,
            pin_memory=self.train_pin_memory,
            drop_last=True,
            shuffle=True,
            collate_fn=self.collate_masks_or_list_of_masks,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.validation_batch_size,
            num_workers=self.validation_num_workers,
            pin_memory=self.validation_pin_memory,
            shuffle=False,
            collate_fn=self.collate_masks_or_list_of_masks,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.validation_batch_size,
            num_workers=self.validation_num_workers,
            pin_memory=self.validation_pin_memory,
            shuffle=False,
        )
