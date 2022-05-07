import os
import wandb

from crownns.main import crowNNs
from pytorch_lightning.loggers import WandbLogger

from config import *

wandb.init(
    project=WANDB_PROJECT_NAME,
    entity=WANDB_ENTITY
)
wandb_logger = WandbLogger()


def build_model():
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

    return m

if __name__ == "__main__":
    model = build_model()

    # Start training
    model.trainer.fit(model)
