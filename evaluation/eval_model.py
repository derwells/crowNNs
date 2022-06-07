import os
import argparse
import pandas as pd

from deepforest import evaluate
from crownns.main import crowNNs
from evaluation.helpers import *
from config import *

SCORE_THRESH = 0.3


def eval(mfile):
    """Evaluate the training set using the model."""

    target_csv = EVAL_CSV
    root_dir = EVAL_ROOT

    m = crowNNs().load_from_checkpoint(mfile)
    m.config["score_thresh"] = SCORE_THRESH
    m.freeze()

    predictions = m.predict_file(csv_file=target_csv, root_dir=root_dir)
    predictions = add_site_field(predictions)

    ground_truth = pd.read_csv(target_csv)

    result = evaluate.evaluate(
        predictions=predictions, ground_df=ground_truth, root_dir=root_dir
    )

    result = result["results"]
    result = add_site_field(result)
    result["match"] = result.IoU > 0.4

    sites = result["site"].unique()
    print(len(sites))
    for site in sites:

        r_site = result[result["site"] == site]
        p_site = predictions[predictions["site"] == site]

        if r_site.shape[0] == 0 or p_site.shape[0]:
            print(f"No preds for {site}")
            continue

        true_positive = sum(r_site["match"])
        recall = true_positive / r_site.shape[0]
        precision = true_positive / p_site.shape[0]

        if precision >= 0.55 and recall >= 0.5:
            print(site)
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
