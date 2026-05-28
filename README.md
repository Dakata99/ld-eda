# Experimental machine learning study for liver disease classification

This project aims to evaluate a portfolio of machine learning algorithms for liver diseases classification.
The experimental study consists of 3 experimental scenarios:

1) Multiclass classification of the largest dataset
2) Binary classification of the largest dataset
3) Binary classification of all 3 datasets

## Prerequisites

```bash
sudo snap install astral-uv --classic
```

## Datasets

The following datasets are being used:
- [indian-liver-disease-dataset](https://www.kaggle.com/datasets/paramjeetsinghds/indian-liver-disease-dataset)
- [hcv-data](https://www.kaggle.com/datasets/visheshkkl/hcv-data)
- [liver-data](https://www.kaggle.com/datasets/aichamalouche/liver-data)

## How to run?

To evaluate the experiments, first set up the environment:
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

Configuration files are located in the `configs` folder.
The structure is as follows:
```json
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
    "random-forest": {...},
    "svm": {...},
    "gradient-boosting": {...},
    "tree": {...},
    "neural-network": {...}
}
```
where the parameters for each learner are added.

### Mapping between Orange's GUI and Python API

#### Logistic regression

| Field               | Type            | Orange's equivalent         | Python's default | Orange's default |
|---------------------|-----------------|-----------------------------|------------------|------------------|
| `penalty`           | list(str\|null) | Regularization type.        | `l2`             | `l2`             |
| `C`                 | list(float)     | Strength (weak to strong).  | `1.0`            | `1.0`            |
| `class_weight`      | list(str\|null) | Balance class distribution. | `None`           | `None`           |

Additional arguments, not exposed by Orange GUI:

| Field               | Python's default | Orange's default |
|---------------------|------------------|------------------|
| `dual`              | `False`          | `False`          |
| `tol`               | `0.0001`         | `0.0001`         |
| `random_state`      | `None`           | `0`              |
| `max_iter`          | `100`            | `10000`          |
| `fit_intercept`     | `True`           | `True`           |
| `intercept_scaling` | `1`              | `1.0`            |

> NOTE: `class_weight` parameter accepts `None` or `balanced` values.

For reference: `Orange/widgets/model/owlogisticregression.py`.

#### Random forest

| Field               | Type                     | Orange's equivalent                            | Python's default | Orange's default               |
|---------------------|--------------------------|------------------------------------------------|------------------|--------------------------------|
| `n_estimators`      | list(int)                | Number of trees.                               | `10`             | `10`                           |
| `max_features`      | list(int \| str \| null) | Number of attributes considered at each split. | `sqrt`           | Disabled: `sqrt`, enabled: `5` |
| `random_state`      | list(int \| null)        | Replicable training.                           | `None`           | Disabled: `None`, enabled: `0` |
| `class_weight`      | list(str \| null)        | Balance class distribution.                    | `None`           | `None`                         |
| `max_depth`         | list(int \| null)        | Limit depth of individual trees.               | `None`           | Disabled: `None`, enabled: `3` |
| `min_samples_split` | list(int)                | Do not split subsets smaller than.             | `2`              | `5`                            |

For reference: `Orange/widgets/model/owrandomforest.py`.

#### Tree

| Field                 | Type              | Orange's equivalent                 | Python's default | Orange's default |
|-----------------------|-------------------|-------------------------------------|------------------|------------------|
| `binarize`            | list(bool)        | Induce binary tree.                 | `False`          | `True`           |
| `min_samples_leaf`    | list(int)         | Min. number of instances in leaves. | `1`              | `2`              |
| `min_samples_split`   | list(int)         | Do not split subsets smaller than.  | `2`              | `5`              |
| `max_depth`           | list(int \| null) | Limit maximal tree depth.           | `None`           | `100`            |
| `sufficient_majority` | list(float)       | Stop when majority reaches.         | `0.95`           | `0.95`           |

For reference: `Orange/widgets/model/owtree.py`.

#### Gradient boosting

| Field               | Type              | Orange's equivalent                | Python's default | Orange's default |
|---------------------|-------------------|------------------------------------|------------------|------------------|
| -                   | -                 | Method.                            | -                | `GBLearner`      |
| `n_estimators`      | list(int)         | Number of trees.                   | `100`            | `100`            |
| `learning_rate`     | list(float)       | Learning rate.                     | `0.1`            | `0.1`            |
| `random_state`      | list(int \| null) | Replicable training.               | `None`           | `0`              |
| `max_depth`         | list(int)         | Limit depth of individual trees.   | `3`              | `3`              |
| `min_samples_split` | list(int)         | Do not split subsets smaller than. | `2`              | `2`              |
| `subsample`         | list(float)       | Fraction of training instances.    | `1.0`            | `1`              |

> NOTE: This table describes the standard `GBLearner`. Other methods, such as `XGBLearner`, `XGBRFLearner`, and `CatGBLearner`, are selected through different learner classes and may have different default parameters.

For reference: `Orange/widgets/model/owgradientboosting.py`.

#### Neural network

| Field                | Type              | Orange's equivalent           | Python's default | Orange's default |
|----------------------|-------------------|-------------------------------|------------------|------------------|
| `hidden_layer_sizes` | list(list(int))   | Neurons in hidden layers.     | `(100,)`         | `(100,)`         |
| `activation`         | list(str)         | Activation.                   | `relu`           | `relu`           |
| `solver`             | list(str)         | Solver.                       | `adam`           | `adam`           |
| `alpha`              | list(float)       | Regularization, alpha.        | `0.0001`         | `0.0001`         |
| `max_iter`           | list(int)         | Maximal number of iterations. | `200`            | `200`            |
| `random_state`       | list(int \| null) | Replicable training.          | `None`           | `1`              |

For reference: `Orange/widgets/model/owneuralnetwork.py`.

#### Support vector machine (SVM)

| Field         | Type               | Orange's equivalent  | Python's default | Orange's default | Notes                                   |
|---------------|--------------------|--------------------- |------------------|------------------|-----------------------------------------|
| `C`           | list(float)        | Cost.                | `1.0`            | `1.0`            | Used by C-SVM.                          |
| `kernel`      | list(str)          | Kernel.              | `rbf`            | `rbf`            | `linear`, `poly`, `rbf`, `sigmoid`.     |
| `gamma`       | list(float \| str) | g                    | `auto`           | `auto`           | Used by Polynomial, RBF and Sigmoid.    |
| `coef0`       | list(float)        | c                    | `0.0`            | `1.0`            | Used by Polynomial and Sigmoid kernels. |
| `tol`         | list(float)        | Numerical tolerance. | `0.001`          | `0.001`          |                                         |
| `max_iter`    | list(int)          | Iteration limit.     | `-1`             | `100`            | `-1` means unlimited in the Python API. |
| `degree`      | list(int)          | d                    | `3`              | `3`              | Relevant for polynomial kernel.         |
| `probability` | list(bool)         |                      | `False`          | `True`           | Orange GUI passes this internally.      |

> NOTE: The configuration for this learner uses lists of dicts since different kernels have different parameters.

For reference: `Orange/widgets/model/owsvm.py`.

## Known issues and limitations

- Disabled GUI options should usually be represented by omitting the parameter from the JSON, not by passing `null`. Passing `null` becomes `None` in Python and is only valid for parameters whose API explicitly accepts `None`, such as `class_weight`, `max_depth`, or `random_state`.
- `Orange.evaluation.testing.sample()` uses a different splitting implementation/row-selection logic than Orange GUI’s Data Sampler widget, so the same `n=0.8`, `stratified=True`, and `random_state=42` do not guarantee the same train/test rows.

## TODO

- Make `default` JSON files for Orange and for Python, i.e. `default-ow` and `default-py`.
- Make full default JSON with all parameters described for each learner.
