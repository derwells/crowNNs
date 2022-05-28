import os
import argparse
import pandas as pd

import matplotlib.pyplot as plt
from crownns.main import crowNNs
from config import *

EVAL_CSV = "data/evaluation/RGB/benchmark_annotations.csv"
EVAL_ROOT = "data/evaluation/RGB"
SCORE_THRESH = 0.35


def f1_score(p, r):
    return 2 * p * r / (p + r)


def predict_image(mfile):
    m = crowNNs().load_from_checkpoint(mfile)
    m.config["score_thresh"] = SCORE_THRESH
    m.freeze()

    img = m.predict_image(
        path="data/evaluation/RGB/UKFS_024_2020.tif", return_plot=True
    )

    plt.imshow(img[:, :, ::-1])
    plt.savefig("test.png")


def get_models_to_test(model_dir):
    models_to_test = os.listdir(model_dir)
    models_to_test = [model_dir + e for e in models_to_test]

    return models_to_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", "-d", type=str)
    args = parser.parse_args()

    models_to_test = get_models_to_test(args.model_dir)

    for mfile in models_to_test:
        predict_image(mfile)


