import os


XML_PATH = "../data/annotations/"
TRAIN_PATH = "../data/training/"
TIF_DIR = os.path.join(TRAIN_PATH, "RGB")
ANNOTATIONS_PATH = os.path.join(TIF_DIR, "annotations.csv")

CROP_DIR = os.path.join(TRAIN_PATH, "crop")
TRAIN_ANNOTATIONS_PATH = os.path.join(CROP_DIR, "train.csv")
VAL_ANNOTATIONS_PATH = os.path.join(CROP_DIR, "val.csv")

PATCH_SIZE = 225
PATCH_OVERLAP = 0.05

SCORE_THRESH = 0.3
EPOCHS = 10
