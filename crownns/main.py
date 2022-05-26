import torch
import crownns.model
import crownns.dataset

import deepforest
import deepforest.main
import deepforest.dataset
import deepforest.utilities

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor



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

    def create_trainer(self, logger=None, callbacks=[], **kwargs):
        """Override create_trainer"""
        
        #If val data is passed, monitor learning rate and setup classification metrics
        if not self.config["validation"]["csv_file"] is None:
            if logger is not None:
                lr_monitor = LearningRateMonitor(logging_interval='epoch')
                callbacks.append(lr_monitor)
        
        self.trainer = pl.Trainer(logger=logger,
                                max_epochs=self.config["train"]["epochs"],
                                gpus=self.config["gpus"],
                                enable_checkpointing=True,
                                accelerator=self.config["distributed_backend"],
                                fast_dev_run=self.config["train"]["fast_dev_run"],
                                callbacks=callbacks,
                                **kwargs)
