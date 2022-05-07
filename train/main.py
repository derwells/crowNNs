import os
import time
import numpy as np
import torch
import helpers

from crownns.main import crowNNs
from deepforest import preprocess
from deepforest import utilities
from pytorch_lightning.loggers import WandbLogger
from config import *

wandb_logger = WandbLogger()


if __name__ == "__main__":
    train_imgs = helpers.get_train_images(TIF_DIR)
    train_xmls = helpers.imgs_to_xml(train_imgs)

    # Build cropped images folder + annotations in CROP_DIR
    for img in train_imgs:
        helpers.preprocess_image(img)

    # Build model
    m = crowNNs()

    # Explicitly use GPU
    m.config['gpus'] = '-1'

    m.config["score_thresh"] = SCORE_THRESH
    m.config["train"]['epochs'] = EPOCHS

    m.config["train"]["csv_file"] = TRAIN_ANNOTATIONS_PATH
    m.config["train"]["root_dir"] = os.path.dirname(TRAIN_ANNOTATIONS_PATH)
    m.config["validation"]["csv_file"] = VAL_ANNOTATIONS_PATH
    m.config["validation"]["root_dir"] = os.path.dirname(VAL_ANNOTATIONS_PATH)

    # Use WanDB logger for PyTorch lighning
    m.create_trainer(logger=wandb_logger)

    # Start training
    m.trainer.fit(m)
