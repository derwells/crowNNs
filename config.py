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

PATCH_SIZE = model_configs['patch_size']
PATCH_OVERLAP = model_configs['patch_overlap']
SCORE_THRESH = model_configs['score_thresh']
EPOCHS = model_configs['epochs']
N_WORKERS = model_configs['n_workers']
BATCH_SIZE = model_configs['batch_size']

# Model output config
MODELS_DIR = configp['output']['model_dir']
