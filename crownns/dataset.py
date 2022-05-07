import os
import pandas as pd
import numpy as np
import torch
import deepforest.dataset

from PIL import Image


class TreeDataset(deepforest.dataset.TreeDataset):
    """Overrides deepforest.dataset.TreeDataset"""

    def __getitem__(self, idx):
        if self.train:
            image_names, image, targets = super().__getitem__(idx)
            targets["boxes"] = targets["boxes"].float()

            return image_names, image, targets

        else:
            converted_image = super().__getitem__(idx)
            converted_image = converted_image.float()

            return converted_image
