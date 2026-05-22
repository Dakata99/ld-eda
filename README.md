# Liver disease explorarory data analysis

This projects aims to show a portfolio of machine learning algorithms for diagnozing liver diseases.
The exploratory data analysis consists of 3 experimental scenarios:

1) Multilabel classification of the biggest dataset
2) Binary classification of the biggest dataset
3) Binary classification of all 3 datasets

## Prerequisites

```bash
sudo snap install astral-uv --classic
```

## Datasets

The following datasets are being used for this EDA:
- [indian-liver-disease-dataset](https://www.kaggle.com/datasets/paramjeetsinghds/indian-liver-disease-dataset)
- [hcv-data](https://www.kaggle.com/datasets/visheshkkl/hcv-data)
- [liver-data](https://www.kaggle.com/datasets/aichamalouche/liver-data)

## How to run?

To evaluate the experiments, firstly setup the environment by:
```bash
source setupenv
```

Then the `liver` command will be present and you can run `liver -h` to see what it does:
```bash
$ liver -h
usage: liver [-h] --experiment {1,2,3} [--debug]
             [--learners-group {logistic-regression,random-forest,tree,gradient-boosting,neural-network,svm} [{logistic-regression,random-forest,tree,gradient-boosting,neural-network,svm} ...]]
             [--config {default,global,experiment1,experiment2,experiment3}] [--plot-only]

options:
  -h, --help            show this help message and exit
  --experiment {1,2,3}
  --debug               Enable debug logging
  --learners-group {logistic-regression,random-forest,tree,gradient-boosting,neural-network,svm} [{logistic-regression,random-forest,tree,gradient-boosting,neural-network,svm} ...]
                        Run specific family(ies) of learners.
  --config {default,global,experiment1,experiment2,experiment3}
                        Configuration to use for the experiment
  --plot-only           Plot only on already existing results.
```

To check the generated results/report, run:
```bash
wslview results/experiment<experiment>-<config>.csv
wslview reports/experiment<experiment>-<config>.html
```

## Configuration files

Configuration files are present at `configs` folder.
The structure is as follows:
```json
{
{
    "logistic-regression": {
        "penalty": [
            "l2"
        ],
        "C": [
            1.0
        ],
        "class_weight": [
            null
        ]
    },
    },
    "random-forest": {...},
    "svm": {...},
    "gradient-boosting": {...},
    "tree": {...},
    "neural-network": {...}
}
```
where the parameters for each learners are added.

### Mapping between Orange's GUI and Python API

#### Logistic regression

| Field          | Type      | Orange's equivalent         |
|----------------|-----------|-----------------------------|
| `penalty`      | list(str) | Regularization type.        |
| `C`            | list(int) | Strength (weak to strong).  |
| `class_weight` | str       | Balance class distribution. |

> NOTE: `class_weight` parameters accepts `None` or `balanced` values.

#### Random forest

| Field               | Type      | Orange's equivalent                            |
|---------------------|-----------|------------------------------------------------|
| `n_estimators`      | list(int) | Number of trees.                               |
| `max_features`      | list(int) | Number of attributes considered at each split. |
| `random_state`      | int       | Replicable training.                           |
| `class_weight`      | str       | Balance class distribution.                    |
| `max_depth`         | list(int) | Limit depth of individual trees.               |
| `min_samples_split` | list(int) | Do not split subsets smaller than.             |

#### Tree

| Field                 | Type      | Orange's equivalent                 |
|-----------------------|-----------|-------------------------------------|
| `binarize`            | boolean   | Induce binary tree.                 |
| `min_samples_leaf`    | string    | Min. number of instances in leaves. |
| `min_samples_split`   | list(int) | Do not split subsets smaller than.  |
| `max_depth`           | list(int) | Limit maximal tree depth.           |
| `sufficient_majority` | list(int) | Stop when majority reaches.         |

#### Gradient boosting

| Field               | Type      | Orange's equivalent                |
|---------------------|-----------|------------------------------------|
| -                   | -         | Method.                            |
| `n_estimators`      | string    | Number of trees.                   |
| `learning_rate`     | list(int) | Learning rate.                     |
| `random_state`      | int       | Replicable training.               |
| `max_depth`         | list(int) | Limit depth of individual trees.   |
| `min_samples_split` | list(int) | Do not split subsets smaller than. |
| `subsample`         | list(int) | Fraction of training instances.    |

> Method can be picked by using different learners: `XGBLearner`, `XGBRFLearner`, `CatGBLearner`.

#### Neural network

| Field                | Type            | Orange's equivalent           |
|----------------------|-----------------|-------------------------------|
| `hidden_layer_sizes` | list(list(int)) | Neurons in hidden layers.     |
| `activation   `      | list(str)       | Activation.                   |
| `solver`             | list(str)       | Solver.                       |
| `alpha`              | list(int)       | Regularization, alpha.        |
| `max_iter`           | list(int)       | Maximal number of iterations. |
| `random_state`       | int             | Replicable training.          |

#### Support machine vectors (SVM)

| Field      | Type        | Orange's equivalent                     | Notes                                   |
|------------|-------------|-----------------------------------------|-----------------------------------------|
| `C`        | list(float) | Cost.                                   |                                         |
| `kernel`   | str         | Kernel.                                 |                                         |
| `gamma`    | list(int)   | g                                       | Gamma (g) for Polynomial kernel.        |
| `coef0`    | list(int)   | -                                       | Used by Polynomial and Sigmoid kernels. |
| `tol`      | list(float) | Numerical tolerance.                    |                                         |
| `max_iter` | list(int)   | Iteration limit.                        |                                         |

> NOTE: the configuration for this learner presents dicts of dicts since different kernels have different parameters.

## Limitations

- It can't be made when a parameter is disabled or not since passing `null` in the JSON will result in `None` value in Python, which may not be correct for some parameters of some learners.
