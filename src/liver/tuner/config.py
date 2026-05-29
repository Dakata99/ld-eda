from __future__ import annotations

from typing import Any

# -----------------------------------------------------------------------------
# Learner/parameter registry
# -----------------------------------------------------------------------------
# Rule:
# - The keys inside "params" are Python API parameter names.
# - "orange_opt" is only the label/name shown in Orange's GUI.
# - Python-only parameters can use orange_opt=None and exposed_in_orange=False.
#
# This lets the GUI display friendly Orange names while the runner still builds
# correct Python kwargs such as {"class_weight": "balanced"}.

LearnerSpecs = dict[str, dict[str, Any]]

LEARNER_SPECS: LearnerSpecs = {
    "LR": {
        "display_name": "Logistic Regression",
        "api_class": "LogisticRegressionLearner",
        "orange_widget": "Logistic Regression",
        "params": {
            "C": {
                "default": 1.0,
                "manual_values": [0.01, 0.05, 0.1, 1.0, 5.0, 10.0],
                "type": "float",
                "orange_opt": "Regularization strength",
                "exposed_in_orange": True,
                "description": "Inverse regularization strength. Smaller values mean stronger regularization.",
            },
            "penalty": {
                "default": "l2",
                "manual_values": ["l2", "l1", None],
                "type": "choice",
                "orange_opt": "Regularization type",
                "exposed_in_orange": True,
                "description": "Regularization penalty passed to the learner.",
            },
            "class_weight": {
                # Orange GUI idea: Balance class distribution = on/off.
                # Python API value should not be True/False for sklearn-like learners;
                # use None or "balanced" instead.
                "default": None,
                "manual_values": [None, "balanced"],
                "type": "choice",
                "orange_opt": "Balance class distribution",
                "exposed_in_orange": True,
                "ui_values": {
                    None: "Disabled",
                    "balanced": "Enabled / balanced",
                },
                "description": "Maps Orange's 'Balance class distribution' option to Python's class_weight parameter.",
            },
            "max_iter": {
                "default": 10000,
                "manual_values": [1000, 5000, 10000],
                "type": "int",
                "orange_opt": None,
                "exposed_in_orange": False,
                "description": "Python API-only safeguard for convergence-heavy Logistic Regression runs.",
            },
        },
    },
    "NN": {
        "display_name": "Neural Network",
        "api_class": "NNClassificationLearner",
        "orange_widget": "Neural Network",
        "params": {
            "hidden_layer_sizes": {
                "default": [100],
                "manual_values": [[100], [50, 50], [100, 50]],
                "type": "list[int]",
                "orange_opt": "Neurons in hidden layers",
                "exposed_in_orange": True,
                "description": "List-like architecture. Manual mode must use a list of lists.",
            },
            "activation": {
                "default": "relu",
                "manual_values": ["relu", "tanh", "logistic"],
                "type": "choice",
                "orange_opt": "Activation",
                "exposed_in_orange": True,
                "description": "Activation function for hidden layers.",
            },
            "alpha": {
                "default": 0.0001,
                "manual_values": [0.0001, 0.001, 0.01],
                "type": "float",
                "orange_opt": "Regularization, α",
                "exposed_in_orange": True,
                "description": "L2 regularization term for the neural network.",
            },
            "max_iter": {
                "default": 200,
                "manual_values": [200, 500, 1000],
                "type": "int",
                "orange_opt": None,
                "exposed_in_orange": False,
                "description": "Python API-only iteration limit.",
            },
        },
    },
    "SVM": {
        "display_name": "SVM",
        "api_class": "SVMLearner",
        "orange_widget": "SVM",
        "params": {
            "C": {
                "default": 1.0,
                "manual_values": [0.1, 1.0, 5.0, 10.0],
                "type": "float",
                "orange_opt": "Cost, C",
                "exposed_in_orange": True,
                "description": "Penalty/cost parameter.",
            },
            "kernel": {
                "default": "rbf",
                "manual_values": ["rbf", "linear", "poly", "sigmoid"],
                "type": "choice",
                "orange_opt": "Kernel",
                "exposed_in_orange": True,
                "description": "Kernel function.",
            },
            "gamma": {
                "default": "auto",
                "manual_values": ["auto", 0.01, 0.1, 1.0],
                "type": "choice|float",
                "orange_opt": "Numerical tolerance / kernel coefficient",
                "exposed_in_orange": True,
                "description": "Kernel coefficient. Keep values conservative; SVM grids can explode quickly.",
            },
        },
    },
    "GB": {
        "display_name": "Gradient Boosting",
        "api_class": "GBClassifier",
        "orange_widget": "Gradient Boosting",
        "params": {
            "n_estimators": {
                "default": 100,
                "manual_values": [50, 100, 200],
                "type": "int",
                "orange_opt": "Number of trees",
                "exposed_in_orange": True,
                "description": "Number of boosting stages.",
            },
            "learning_rate": {
                "default": 0.1,
                "manual_values": [0.03, 0.05, 0.1],
                "type": "float",
                "orange_opt": "Learning rate",
                "exposed_in_orange": True,
                "description": "Shrinkage applied to each tree contribution.",
            },
            "max_depth": {
                "default": 3,
                "manual_values": [2, 3, 5],
                "type": "int",
                "orange_opt": "Limit depth of individual trees",
                "exposed_in_orange": True,
                "description": "Depth of weak learners used by boosting.",
            },
        },
    },
    "Tree": {
        "display_name": "Tree",
        "api_class": "TreeLearner",
        "orange_widget": "Tree",
        "params": {
            "max_depth": {
                "default": None,
                "manual_values": [None, 5, 10, 20],
                "type": "optional[int]",
                "orange_opt": "Limit depth",
                "exposed_in_orange": True,
                "description": "None means no explicit maximum depth.",
            },
            "min_samples_split": {
                "default": 2,
                "manual_values": [2, 5, 10],
                "type": "int",
                "orange_opt": "Do not split subsets smaller than",
                "exposed_in_orange": True,
                "description": "Minimum number of samples required to split an internal node.",
            },
            "min_samples_leaf": {
                "default": 1,
                "manual_values": [1, 3, 5],
                "type": "int",
                "orange_opt": "Stop splitting when majority reaches / min leaf size",
                "exposed_in_orange": True,
                "description": "Minimum samples in a leaf. Helps reduce overfitting.",
            },
            "binarize": {
                "default": False,
                "manual_values": [False, True],
                "type": "bool",
                "orange_opt": "Binarization",
                "exposed_in_orange": False,
                "description": "Python API option. Use only if you explicitly want to test it.",
            },
        },
    },
    "RF": {
        "display_name": "Random Forest",
        "api_class": "RandomForestLearner",
        "orange_widget": "Random Forest",
        "params": {
            "n_estimators": {
                "default": 100,
                "manual_values": [50, 100, 200],
                "type": "int",
                "orange_opt": "Number of trees",
                "exposed_in_orange": True,
                "description": "Number of trees in the forest.",
            },
            "max_features": {
                "default": "sqrt",
                "manual_values": ["sqrt", "log2", None],
                "type": "choice",
                "orange_opt": "Number of attributes considered at each split",
                "exposed_in_orange": True,
                "description": "None usually means all features; sqrt/log2 restrict the split search.",
            },
            "max_depth": {
                "default": None,
                "manual_values": [None, 10, 20],
                "type": "optional[int]",
                "orange_opt": "Limit depth of individual trees",
                "exposed_in_orange": True,
                "description": "None means unconstrained depth.",
            },
            "min_samples_split": {
                "default": 2,
                "manual_values": [2, 5, 10],
                "type": "int",
                "orange_opt": "Do not split subsets smaller than",
                "exposed_in_orange": True,
                "description": "Minimum samples required to split an internal tree node.",
            },
            "class_weight": {
                "default": None,
                "manual_values": [None, "balanced"],
                "type": "choice",
                "orange_opt": "Balance class distribution",
                "exposed_in_orange": True,
                "ui_values": {
                    None: "Disabled",
                    "balanced": "Enabled / balanced",
                },
                "description": "Maps Orange's class balancing option to Python's class_weight parameter.",
            },
        },
    },
}

LEARNER_PAIRS = [
    ("LR", "NN"),
    ("SVM", "GB"),
    ("Tree", "RF"),
]
