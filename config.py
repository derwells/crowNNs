import os
import configparser

configp = configparser.ConfigParser()
configp.read(".config.cfg")

# wandb configs
WANDB_PROJECT_NAME = configp['wandb']['project_name']

# Training/Eval folder configs
XML_PATH = "data/annotations/"
TRAIN_PATH = "data/training/"
TIF_DIR = os.path.join(TRAIN_PATH, "RGB")
ANNOTATIONS_PATH = os.path.join(TIF_DIR, "annotations.csv")

CROP_DIR = os.path.join(TRAIN_PATH, "crop")
TRAIN_ANNOTATIONS_PATH = os.path.join(CROP_DIR, "train.csv")
VAL_ANNOTATIONS_PATH = os.path.join(CROP_DIR, "val.csv")

# Model configs
model_configs = configp['model']

PATCH_SIZE  = configp.getfloat('model', 'patch_size')
PATCH_OVERLAP = configp.getfloat('model', 'patch_overlap')
SCORE_THRESH = configp.getfloat('model', 'score_thresh')
NMS_THRESH = configp.getfloat('model', 'nms_thresh')
EPOCHS = configp.getint('model', 'epochs')
N_WORKERS = configp.getint('model', 'n_workers')
BATCH_SIZE = configp.getint('model', 'batch_size')
LEARNING_RATE = configp.getfloat('model', 'learning_rate')

# Model output config
MODELS_DIR = configp['output']['model_dir']
