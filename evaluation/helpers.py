import os

EVAL_CSV = "data/evaluation/RGB/benchmark_annotations.csv"
EVAL_ROOT = "data/evaluation/RGB"


def get_models_to_test(model_dir):
    models_to_test = os.listdir(model_dir)
    models_to_test = [model_dir + e for e in models_to_test]

    return models_to_test
