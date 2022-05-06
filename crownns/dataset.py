import os
import pandas as pd
import numpy as np
import torch
import deepforest

from PIL import Image


class TreeDataset(deepforest.dataset.TreeDataset):
    """Overrides deepforest.dataset.TreeDataset"""

    def __getitem__(self, idx):
        # Read image if not in memory
        if self.preload_images:
            image = self.image_dict[idx]
        else:
            img_name = os.path.join(self.root_dir, self.image_names[idx])
            image = Image.open(img_name).convert("RGB")
            image = np.array(image) / 255
            image = image.astype("float32")

        if self.train:
            # Select annotations
            image_annotations = self.annotations[
                self.annotations.image_path == self.image_names[idx]
            ]
            targets = {}
            targets["boxes"] = image_annotations[
                ["xmin", "ymin", "xmax", "ymax"]
            ].values.astype("float32")

            # Labels need to be encoded
            targets["labels"] = image_annotations.label.apply(
                lambda x: self.label_dict[x]
            ).values.astype(np.int64)

            # If image has no annotations, don't augment
            if np.sum(targets["boxes"]) == 0:
                boxes = boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.from_numpy(targets["labels"])
                # Channels last
                image = np.rollaxis(image, 2, 0)
                image = torch.from_numpy(image)
                targets = {"boxes": boxes, "labels": labels}
                return self.image_names[idx], image, targets

            augmented = self.transform(
                image=image, bboxes=targets["boxes"], category_ids=targets["labels"]
            )
            image = augmented["image"]

            boxes = np.array(augmented["bboxes"])
            boxes = torch.from_numpy(boxes).float()
            labels = np.array(augmented["category_ids"])
            labels = torch.from_numpy(labels)
            targets = {"boxes": boxes, "labels": labels}

            return self.image_names[idx], image, targets

        else:
            # Mimic the train augmentation
            converted = self.image_converter(image=image)
            converted["image"] = converted["image"].float()

            return converted["image"]
