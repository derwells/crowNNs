import argparse
import matplotlib.pyplot as plt

from crownns.main import crowNNs
from evaluation.helpers import *
from config import *

SCORE_THRESH = 0.35
IMG_PATH = ""

def predict_image(mfile):
    """Predict trees given an images"""

    m = crowNNs().load_from_checkpoint(mfile)
    m.config["score_thresh"] = SCORE_THRESH
    m.freeze()

    img = m.predict_image(
        path=IMG_PATH, return_plot=True
    )

    plt.imshow(img[:, :, ::-1])
    plt.savefig("test.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", "-d", type=str)
    args = parser.parse_args()

    models_to_test = get_models_to_test(args.model_dir)

    for mfile in models_to_test:
        predict_image(mfile)


