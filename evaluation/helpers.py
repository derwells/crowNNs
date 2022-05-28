import os

EVAL_CSV = "data/evaluation/RGB/benchmark_annotations.csv"
EVAL_ROOT = "data/evaluation/RGB"


def get_models_to_test(model_dir):
    models_to_test = os.listdir(model_dir)
    models_to_test = [
        os.path.join(model_dir, m_file) for m_file in models_to_test
    ]

    return models_to_test


def f1_score(p, r):
    return 2 * p * r / (p + r)

