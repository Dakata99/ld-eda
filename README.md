# Liver disease explorarory data analysis

## Prerequisites

```bash
sudo snap install astral-uv --classic
```

## How to run?

To evaluate the experiments, firstly setup the environment by:
```bash
source setupenv
```

Then the `liver` command will be present and you can run `liver -h` to see what it does:
```bash
$ liver -h
usage: liver [-h] --experiment {1,2,3} [--debug] [--learner-group {all,logistic-regression,random-forest,tree,gradient-boosting,neural-network,svm}]
             [--config {global,experiment1,experiment2,experiment3}] [--plot-only]

options:
  -h, --help            show this help message and exit
  --experiment {1,2,3}
  --debug               Enable debug logging
  --learner-group {all,logistic-regression,random-forest,tree,gradient-boosting,neural-network,svm}
  --config {global,experiment1,experiment2,experiment3}
                        Configuration to use for the experiment
  --plot-only           Plot only on already existing results.
```

## Configuration files

Configuration files are present at `configs` folder.
The structure is as follows:
```json
{
    "logistic-regression": {
        "penalty": [
            "l2",
            "l1"
        ],
        "C": [
            0.1,
            1,
            5,
            10
        ],
        "class_weight": ["balanced"]
    },
    "random-forest": {
        ...
    },
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

| Field      | Type        | Orange's equivalent                     |
|------------|-------------|-----------------------------------------|
| `C`        | list(float) | Cost.                                   |
| `kernel`   | str         | Kernel.                                 |
| `gamma`    | list(int)   | Gamma (g) for Polynomial kernel.        |
| `coef0`    | list(int)   | Used by Polynomial and Sigmoid kernels. |
| `tol`      | list(float) | Numerical tolerance.                    |
| `max_iter` | list(int)   | Iteration limit.                        |

> NOTE: the configuration for this learner presents dicts of dicts since different kernels have different parameters.
