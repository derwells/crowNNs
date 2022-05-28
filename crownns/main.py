import torch
import crownns.model
import crownns.dataset

import deepforest
import deepforest.main
import deepforest.dataset
import deepforest.utilities

import pytorch_lightning as pl

from pytorch_lightning.callbacks import LearningRateMonitor
from torch import optim



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


    def configure_optimizers(self):
        optimizer = optim.SGD(self.model.parameters(),
                                   lr=self.config["train"]["lr"],
                                   momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                            mode='min',
                                                            factor=0.1,
                                                            patience=10,
                                                            verbose=True,
                                                            threshold=0.0001,
                                                            threshold_mode='rel',
                                                            cooldown=0,
                                                            min_lr=0,
                                                            eps=1e-08)

        #Monitor rate is val data is used
        if self.config["validation"]["csv_file"] is not None:
            return {'optimizer':optimizer, 'lr_scheduler': scheduler,"monitor":'val_classification'}
        else:
            return optimizer


    def training_step(self, batch, batch_idx):
        """Train on a loaded dataset
        """
        # Confirm model is in train mode
        self.model.train()
        
        # Allow for empty data if data augmentation is generated
        path, images, targets = batch

        loss_dict = self.model.forward(images, targets)

        # sum of regression and classification loss
        losses = sum([loss for loss in loss_dict.values()])

        # Log training loss
        for key, value in loss_dict.items():
            self.log("loss_{}".format(key), value, on_step=True)     

        return losses
