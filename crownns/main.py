import torch
import crownns
import crownns.model
import crownns.dataset

import deepforest
import deepforest.main
import deepforest.dataset
import deepforest.utilities


class crowNNs(deepforest.main.deepforest):
    """
    CrowNNs main class
    """

    def create_model(self):
        """Override create_model"""

        self.model = crownns.model.create_model(
            self.num_classes, self.config["nms_thresh"], self.config["score_thresh"]
        )

    def load_dataset(
        self,
        csv_file,
        root_dir=None,
        augment=False,
        shuffle=True,
        batch_size=1,
        train=False,
    ):
        """Override load_dataset"""

        ds = crownns.dataset.TreeDataset(
            csv_file=csv_file,
            root_dir=root_dir,
            transforms=self.transforms(augment=augment),
            label_dict=self.label_dict,
            preload_images=self.config["train"]["preload_images"],
        )

        data_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=deepforest.utilities.collate_fn,
            num_workers=self.config["workers"],
        )

        return data_loader
