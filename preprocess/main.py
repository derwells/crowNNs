import os
import preprocess.helpers
import wandb

from pytorch_lightning.loggers import WandbLogger
from config import *

wandb.init(
    project=WANDB_PROJECT_NAME,
    entity=WANDB_ENTITY
)
wandb_logger = WandbLogger()


def do_preprocessing():
    # Remove previously-generated files
    preprocess.helpers.remove_files([
        ANNOTATIONS_PATH,
        TRAIN_ANNOTATIONS_PATH,
        VAL_ANNOTATIONS_PATH
    ])
    os.rmdir(CROP_DIR)

    train_imgs = preprocess.helpers.get_train_images(TIF_DIR)
    train_xmls = preprocess.helpers.imgs_to_xml(train_imgs)

    # Build compiled annotations
    preprocess.helpers.compile_annotations(train_xmls)

    # Build cropped images folder + annotations in CROP_DIR
    for img in train_imgs:
        preprocess.helpers.preprocess_image(img)


if __name__ == "__main__":
    do_preprocessing()
