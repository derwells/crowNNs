import argparse
import pandas as pd

from deepforest import visualize
from crownns.main import crowNNs
from evaluation.helpers import *
from config import *

SCORE_THRESH = 0.325


def f1_score(p, r):
    return 2 * p * r / (p + r)


def eval_img(mfile, save_dir):
    target_csv_path = EVAL_CSV
    root_dir = EVAL_ROOT

    m = crowNNs().load_from_checkpoint(mfile)
    m.config["score_thresh"] = SCORE_THRESH
    m.freeze()

    predictions = m.predict_file(csv_file=target_csv_path, root_dir=root_dir)

    ground_truth = pd.read_csv(target_csv_path)

    files = visualize.plot_prediction_dataframe(
        predictions, root_dir, ground_truth=ground_truth, savedir=save_dir
    )
    print(files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", "-d", type=str)
    parser.add_argument("--save_dir", "-s", type=str)
    args = parser.parse_args()

    models_to_test = get_models_to_test(args.model_dir)

    for mfile in models_to_test:
        eval_img(mfile, args.save_dir)

