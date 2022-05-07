import os
import wandb

from crownns.main import crowNNs
from pytorch_lightning.loggers import WandbLogger

from config import *


if __name__ == "__main__":
    wandb.init(project=WANDB_PROJECT_NAME)
    wandb_logger = WandbLogger()

    # Build model
    m = crowNNs()

    # Explicitly use GPU
    m.config["gpus"] = "-1"

    m.config["score_thresh"] = SCORE_THRESH
    m.config["train"]["epochs"] = EPOCHS
    m.config["workers"] = N_WORKERS
    m.config["batch_size"] = BATCH_SIZE

    m.config["train"]["csv_file"] = TRAIN_ANNOTATIONS_PATH
    m.config["train"]["root_dir"] = os.path.dirname(TRAIN_ANNOTATIONS_PATH)
    m.config["validation"]["csv_file"] = VAL_ANNOTATIONS_PATH
    m.config["validation"]["root_dir"] = os.path.dirname(VAL_ANNOTATIONS_PATH)

    # Use WanDB logger for PyTorch lighning
    m.create_trainer(logger=wandb_logger)

    # Start training
    m.trainer.fit(m)
