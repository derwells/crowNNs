import os
import argparse
import pandas as pd

from deepforest import evaluate
from crownns.main import crowNNs
from evaluation.helpers import *
from config import *

SCORE_THRESH = 0.3


def eval(mfile):
    target_csv = EVAL_CSV
    root_dir = EVAL_ROOT

    m = crowNNs().load_from_checkpoint(mfile)
    m.config["score_thresh"] = SCORE_THRESH
    m.freeze()

    predictions = m.predict_file(csv_file=target_csv, root_dir=root_dir)

    ground_truth = pd.read_csv(target_csv)

    result = evaluate.evaluate(
        predictions=predictions, ground_df=ground_truth, root_dir=root_dir
    )

    result = result["results"]
    result["match"] = result.IoU > 0.4
    true_positive = sum(result["match"])
    recall = true_positive / result.shape[0]
    precision = true_positive / predictions.shape[0]

    print(precision, recall)
    print(f1_score(precision, recall))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", "-d", type=str)
    args = parser.parse_args()

    models_to_test = get_models_to_test(args.model_dir)

    for mfile in models_to_test:
        print(mfile)
        eval(mfile)


