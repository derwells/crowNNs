# CrowNNs

Quick attempt at replacing RetinaNet with FCOS in the [DeepForest](https://github.com/weecology/DeepForest) library.

Uses the same [NeonTreeEvaluation dataset](https://zenodo.org/record/5914554) for benchmarking.

# Results

Trained on an NVIDIA T4 using GCP:

|       Library       | Precision | Recall |
|:-------------------:|:---------:|--------|
|       CrowNNs       |    0.64   |  0.43  |
|  DeepForest (SOTA)  |    0.66   |  0.79  |
| lidR package (2016) |    0.34   |  0.47  |

# Getting Started

Make sure CUDA is installed.

Create a virtual environment using

```
python3 -m venv .venv
```

and install `requirements.txt`.

# Training

```
python3 -m train.main <args>
```

# Evaluation

```
python3 -m evaluation.<script>.py
```
