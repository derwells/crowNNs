import argparse
import pandas as pd

from deepforest import evaluate
from deepforest import visualize
from crownns.main import crowNNs
from config import *

EVAL_CSV = "data/evaluation/RGB/benchmark_annotations.csv"
EVAL_ROOT = "data/evaluation/RGB"


def f1_score(p, r):
    return 2 * p * r / (p + r)


def eval_img(mfile):
    target_csv_path = EVAL_CSV
    root_dir = EVAL_ROOT

    m = crowNNs().load_from_checkpoint(mfile)
    m.config["score_thresh"] = 0.325
    m.freeze()

    predictions = m.predict_file(csv_file=target_csv_path, root_dir=root_dir)

    ground_truth = pd.read_csv(target_csv_path)

    result = evaluate.evaluate(
        predictions=predictions,
        ground_df=ground_truth,
        root_dir=root_dir,
    )

    files = visualize.plot_prediction_dataframe(
        predictions, root_dir, ground_truth=ground_truth, savedir="temp"
    )
    print(files)


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
        eval_img(mfile)

