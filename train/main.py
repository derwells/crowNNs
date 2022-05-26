import os
import wandb
import torch

from crownns.main import crowNNs
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from config import *
import gc
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nms', '-n', type=float)
parser.add_argument('--lr', '-l', type=float)
parser.add_argument('--epochs', '-e', type=int)
parser.add_argument('--dir', '-d', type=str)
parser.add_argument('--skip', '-s', type=bool)
parser.add_argument('--checkpoint', '-c', type=str)
args = parser.parse_args()


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()

    if args.skip == True:
        exit()
    wandb.init(project=WANDB_PROJECT_NAME)
    wandb_logger = WandbLogger()

    # Build model
    if args.checkpoint is not None:
        m = crowNNs().load_from_checkpoint(args.checkpoint)
    else:
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

    m.config["train"]["lr"] = LEARNING_RATE


    if args.nms is not None:
        m.config["nms_thresh"] = args.nms
    if args.lr is not None:
        m.config["train"]["lr"] = args.lr
    if args.epochs is not None:
        m.config["train"]["epochs"] = args.epochs

    dir = f'nms={m.config["nms_thresh"]}-lr={m.config["train"]["lr"]}-e={m.config["train"]["epochs"]}'
    if args.dir is not None:
        dir = args.dir
    os.mkdir(dir)

    callback = ModelCheckpoint(
        dirpath=dir,
        filename='box_precision-{epoch}',
        monitor='box_recall', 
        mode="max",
        save_top_k=-1
    )

    # Use WanDB logger for PyTorch lighning
    m.create_trainer(logger=wandb_logger, callbacks=[callback])

    # Start training
    m.trainer.fit(m)

    # After training
    model_path = "{}/crownns-fcos-resnet50.pl".format(MODELS_DIR)

    # Save checkpoint
    m.trainer.save_checkpoint(model_path)

    # Save model
    torch.save(m.model.state_dict(), model_path)
