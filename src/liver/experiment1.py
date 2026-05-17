"""
TODO: add docstring.
Multiclass classification for indian dataset.
"""

import json
import numpy as np
from sklearn.utils.multiclass import type_of_target
import pandas as pd
from Orange.data.pandas_compat import table_to_frame
from loguru import logger
from itertools import product

from Orange.data import (
    Domain,
    Table,
    DiscreteVariable
)
from Orange.preprocess import (
    PreprocessorList,
    Impute,
    Average,
    Continuize,
    Normalize
)
from Orange.classification import (
    LogisticRegressionLearner,
    RandomForestLearner,
    TreeLearner,
    GBClassifier,
    NNClassificationLearner,
    SVMLearner,
)
from Orange.evaluation.testing import CrossValidation, TestOnTestData, sample
from Orange.evaluation import (
    Recall,
    F1,
    Precision,
    CA,
    AUC,
    MatthewsCorrCoefficient
)

from .load import load_csv, load_configuration

def transform(data):
    """TODO: add docstring."""
    target = data.domain["Liver_Disease_Type"]

    assert type(target) == DiscreteVariable, "Target variable must be discrete!"
    logger.debug(data.domain)

    features = [
        attr for attr in data.domain.attributes
        if attr.name != target.name
    ]
    domain = Domain(
        attributes=features,
        class_vars=target,
        metas=data.domain.metas
    )

    # Transform to new domain
    data = data.transform(domain)

    info = {
        "target": str(data.domain.class_var),
        "labels": list(data.domain.class_var.values),
        "rows": len(data),
        "features": len(data.domain.attributes),
        "attributes": [str(a) for a in data.domain.attributes],
    }

    logger.debug("Dataset info:\n{}", json.dumps(info, indent=2))
    logger.debug(f"Class variable: {data.domain.class_var}")
    logger.debug(f"Class variable type: {type(data.domain.class_var)}")
    logger.debug(f"Is discrete: {data.domain.class_var.is_discrete}")
    logger.debug(f"Is continuous: {data.domain.class_var.is_continuous}")
    logger.debug(f"Y dtype: {data.Y.dtype}")
    logger.debug(f"Unique Y values: {np.unique(data.Y)[:20]}")

    return data


def get_combinations(params):
    """TODO: add docstring."""
    keys = list(params.keys())
    values = list(params.values())

    return [dict(zip(keys, combo)) for combo in product(*values)]


def create_learners(config):
    """TODO: add docstring."""
    # 1) Logistic regression
    configuration = config["logistic-regression"]
    logger.info(configuration)

    combos = get_combinations(configuration)
    logger.debug(f"Logistic regression combinations: {len(combos)}")
    for combo in combos:
        combo["class_weight"] = 'balanced'
        logger.debug(combo)

    logistic_regressions = [LogisticRegressionLearner(**combo) for combo in combos]
    logger.debug(logistic_regressions)

    # 2) Random forest
    configuration = config["random-forest"]
    combos = get_combinations(configuration)
    logger.debug(f"Random forest combinations: {len(combos)}")
    for combo in combos:
        combo["class_weight"] = 'balanced'
        logger.debug(combo)
    random_forests = [RandomForestLearner(**combo) for combo in combos]
    logger.debug(random_forests)

    # 3) Tree
    configuration = config["tree"]
    combos = get_combinations(configuration)
    logger.debug(f"Tree combinations: {len(combos)}")
    for combo in combos:
        logger.debug(combo)
    trees = [TreeLearner(**combo) for combo in combos]
    logger.debug(trees)

    # 4) Gradient boosting
    configuration = config["gradient-boosting"]
    combos = get_combinations(configuration)
    logger.debug(f"Gradient boosting combinations: {len(combos)}")
    for combo in combos:
        logger.debug(combo)
    gradient_boostings = [GBClassifier(**combo) for combo in combos]
    logger.debug(gradient_boostings)

    # 5) Neural network
    configuration = config["neural-network"]
    combos = get_combinations(configuration)
    logger.debug(f"Neural network combinations: {len(combos)}")
    for combo in combos:
        logger.debug(combo)
    neural_networks = [NNClassificationLearner(**combo) for combo in combos]
    logger.debug(neural_networks)

    # 6) SVM
    # FIXME: kerner issues
    configuration = config["svm"]
    combos = get_combinations(configuration)
    logger.debug(f"SVM combinations: {len(combos)}")
    for combo in combos:
        logger.debug(combo)
    svms = [SVMLearner(**combo) for combo in combos]
    logger.debug(svms)

    return {
        "logistic-regression": logistic_regressions,
        "random-forest": random_forests,
        "tree": trees,
        "gradient-boosting": gradient_boostings,
        "neural-network": neural_networks,
        "svm": svms,
    }


def evaluate(data, learners: list, preprocessor):
    """TODO: add docstring."""
    # Split the data into train and test sets (80% train, 20% test, stratified, random state for reproducibility)
    train, test = sample(data, n=0.8, stratified=True, random_state=42)

    # Evaluate using TestOnTestData (train on train set, test on test set)
    scores = TestOnTestData(
        train_data=train,
        test_data=test,
        learners=learners,
        preprocessor=preprocessor,
        store_data=True
    )
    # scores = CrossValidation(data, learners["logistic-regression"], k=5)
    logger.debug(scores)

    # TODO: check if this is valuable
    # # This adds predictions, probabilities, and fold index to a Table
    # augmented = scores.get_augmented_data(
    #     # model_names=learners.keys(),
    #     model_names=[learner.name for learner in learners],
    #     include_attrs=True,
    #     include_predictions=True,
    #     include_probabilities=True,
    # )
    # df = table_to_frame(augmented)
    # logger.info(df.head())
    # Save like any Orange Table
    # augmented.save("augmented_predictions.csv")

    logger.error("Actual shape: {}", scores.actual.shape)
    logger.error("Actual type_of_target: {}", type_of_target(scores.actual))
    logger.error("Actual unique: {}", np.unique(scores.actual))

    logger.error("Predicted full shape: {}", scores.predicted.shape)

    for i, learner in enumerate(learners):
        y_pred = scores.predicted[i]

        logger.error("=" * 80)
        logger.error("Learner: {}", repr(learner))
        logger.error("Prediction shape: {}", y_pred.shape)
        logger.error("Prediction dtype: {}", y_pred.dtype)
        logger.error("Prediction type_of_target: {}", type_of_target(y_pred))
        logger.error("Prediction first 20: {}", y_pred[:20])
        logger.error("Prediction unique first 30: {}", np.unique(y_pred)[:30])

        is_integer_like = np.all(np.isclose(y_pred, np.rint(y_pred)))
        logger.error("Prediction is integer-like: {}", is_integer_like)

    # logger.info("CA:", CA(scores))
    # logger.info("AUC:", AUC(scores))
    # logger.info("F1:", F1(scores, average='weighted'))
    # logger.info("Precision:", Precision(scores, average='weighted'))
    # logger.info("Recall:", Recall(scores, average='weighted'))

    metric_functions = {
        "CA": CA,
        "AUC": AUC,
        "Precision_weighted": lambda res: Precision(scores, average="weighted"),
        "Recall_weighted": lambda res: Recall(scores, average="weighted"),
        "F1_weighted": lambda res: F1(scores, average="weighted"),
        "MCC": MatthewsCorrCoefficient,
    }

    rows = []

    for i, learner in enumerate(learners):
        row = {
            "Learner": repr(learner)
        }

        for metric_name, metric_func in metric_functions.items():
            values = metric_func(scores)
            row[metric_name] = values[i]

        rows.append(row)

    df = pd.DataFrame(rows)

    print(df.to_string(index=False))


def main():
    """TODO: write docstring."""
    logger.info('Running experiment 1: multiclass classification for indian dataset')

    # 1) Load the dataset (CSV file)
    data = load_csv('indian')

    # 2) Do transformation
    data = transform(data)

    # DEBUG: Check target variable type
    logger.warning(f"Target variable: {data.domain.class_var}")
    logger.warning(f"Target type: {type(data.domain.class_var)}")
    logger.warning(f"Target values: {data.domain.class_var.values if hasattr(data.domain.class_var, 'values') else 'N/A'}")
    logger.warning(f"Y dtype: {data.Y.dtype}")
    logger.warning(f"Y unique values: {set(data.Y.ravel())}")

    # 3) Load configuration and create learners
    config = load_configuration()
    learners = create_learners(config)

    logger.debug(learners)

    # 4) Split

    # 5) Preprocess
    preprocessor = PreprocessorList(
        preprocessors=(
            # Average/Most frequent
            Impute(method=Average()),
            # One-hot encoding/One feature per value
            Continuize(multinomial_treatment=Continuize.Indicators),
            # Standardization (z-score normalization)
            Normalize(norm_type=Normalize.NormalizeBySD)
        )
    )

    # TODO: somehow verify that the preprocessor is working correctly (e.g. check for NaNs, check that features are continuous, etc.)
    # data = preprocessor(data)

    # 6) Evaluate
    evaluate(
        data,
        learners["logistic-regression"],# +
        # learners["random-forest"] +
        # learners["tree"] +
        # learners["gradient-boosting"] +
        # learners["neural-network"] +
        # learners["svm"],
        preprocessor
    )
