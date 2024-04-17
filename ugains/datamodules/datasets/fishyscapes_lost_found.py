from pathlib import Path

import numpy as np

from ugains.utils.data_utils import SharedData
from PIL import Image
from torch.utils.data import Dataset


class FishyscapesLostFoundDataset(Dataset):
    id2trainid = np.array([0, 1, 255], dtype="uint8")  # void
    trainid2color = np.concatenate(
        (
            np.array([(144, 238, 144), (255, 102, 102)], dtype="uint8"),
            np.zeros((254, 3), dtype="uint8"),
        )
    )
    trainid2name = np.concatenate(
        (np.array(("background", "anomaly")), np.full(254, "void"))
    )

    def __init__(self, root, transform, target_type="instance", mode="*"):
        self.root = root
        self.target_type = target_type
        self._transform = transform
        self.mode = mode

        targets_base = Path(self.root) / "gtCoarse"
        targets = sorted(list(targets_base.glob(f"{mode}/*/*labelTrainIds.png")))
        images = [
            str(label_path.absolute())
            .replace("gtCoarse_labelTrainIds", "leftImg8bit")
            .replace(r"gtCoarse", "leftImg8bit")
            for label_path in targets
        ]
        # self.target_type = "instance"
        # if "instance" in self.target_type:
        instance_targets = [
            str(label_path.absolute())
            .replace("lost_found", "lost_found_instance")
            .replace("labelTrainIds", "instanceIds")
            for label_path in targets
        ]
        panoptic_targets = [
            str(label_path.absolute())
            .replace("gtCoarse", "gtCoarse_panoptic")
            .replace("lost_found", "lost_found_instance")
            .replace("_labelTrainIds", "")
            for label_path in targets
        ]
        self.instance_targets = SharedData(instance_targets)
        self.targets = SharedData(targets)
        self.images = SharedData(images)
        self.panoptic_targets = SharedData(panoptic_targets)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.targets[index]).convert("L")

        if self._transform is not None:
            target = np.array(target)
            if "data/lost_found" in str(self.targets[index]):
                # working with the case of lost_found, where we have a shifted label
                target[target == 0] = 2
                target[target < 3] = target[target < 3] - 1
            else:
                target = self.id2trainid[target]
            # transformed = self._transform(image=np.array(image), mask=target)
            transformed = self._transform(image=np.array(image))
            image = transformed["image"].transpose(2, 0, 1)

        return_target = {
            "sem_seg": target,
            "id": self.instance_targets[index],
            "panoptic_id": self.panoptic_targets[index],
        }
        return image, return_target


class FishyscapesStaticDataset(FishyscapesLostFoundDataset):
    def __init__(self, root, transform, target_type="semantic"):
        self.root = root
        targets_base = Path(self.root)
        self._transform = transform
        self.targets = sorted(list(targets_base.glob("*labels.png")))
        self.images = [
            str(label_path).replace("labels.png", "rgb.jpg")
            for label_path in self.targets
        ]
        self.target_type = target_type
        self.id2trainid = np.arange(256, dtype="uint8")  # just an identity mapping
