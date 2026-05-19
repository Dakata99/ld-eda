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
from functools import partial

from Orange.data import Domain, Table, DiscreteVariable
from Orange.preprocess import PreprocessorList, Impute, Average, Continuize, Normalize
from Orange.evaluation.testing import CrossValidation, TestOnTestData, sample
from Orange.evaluation import Recall, F1, Precision, CA, AUC, MatthewsCorrCoefficient

from .load import load_dataset, load_configuration
from .utils import create_learners, dataset_report, profiler


def transform(data):
    """TODO: add docstring."""
    target = data.domain["Liver_Disease_Type"]

    assert isinstance(target, DiscreteVariable), "Target variable must be discrete!"
    logger.debug(data.domain)

    features = [attr for attr in data.domain.attributes if attr.name != target.name]
    domain = Domain(attributes=features, class_vars=target, metas=data.domain.metas)

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


@profiler
def evaluate(data, learners: list, preprocessor):
    """TODO: add docstring."""
    # Split the data into train and test sets (80% train, 20% test, stratified, random state for reproducibility)
    train, test = sample(data, n=0.8, stratified=True, random_state=42)

    def progress_callback(progress: float) -> None:
        done = round(progress * len(learners))
        logger.info(
            "Finished {}/{} learners ({:.2f}%)",
            done,
            len(learners),
            progress * 100,
        )

    # Evaluate using TestOnTestData (train on train set, test on test set)
    evaluator = TestOnTestData(
        store_data=False
    )  # set store_data to True if we want to keep the augmented data with predictions, probabilities, etc.
    # CrossValidation(store_data=False, k=5)
    scores = evaluator(
        train,
        test,
        learners,
        preprocessor=preprocessor,
        callback=progress_callback,
    )

    logger.debug(f"CA: {CA(scores)}")
    logger.debug(f"AUC: {AUC(scores)}")
    logger.debug(f"F1: {F1(scores, average='weighted')}")
    logger.debug(f"Precision: {Precision(scores, average='weighted')}")
    logger.debug(f"Recall: {Recall(scores, average='weighted')}")

    # TODO: decide which average to use for multiclass classification (e.g. weighted, macro, micro)
    metrics = {
        "CA": CA,
        "AUC": AUC,
        "Precision(average=weighted)": partial(Precision, average="weighted"),
        "Recall(average=weighted)": partial(Recall, average="weighted"),
        "F1(average=weighted)": partial(F1, average="weighted"),
        "MCC": MatthewsCorrCoefficient,
    }

    rows = []

    # Loop through learners
    for i, learner in enumerate(learners):
        row = {"Learner": repr(learner)}

        for name, metric in metrics.items():
            values = metric(scores)
            row[name] = values[i]

        rows.append(row)

    df = pd.DataFrame(rows)
    logger.success(df.to_string(index=False))
    df.to_csv("evaluation_results.csv", index=False)


def main():
    """TODO: write docstring."""
    logger.info("Running experiment 1: multiclass classification for indian dataset")

    # 1) Load the dataset (CSV file)
    data = load_dataset("expr1")
    dataset_report(
        data,
        preview_rows=10,
        show_all_features=True,
        show_all_metas=True,
        show_numeric_stats=True,
        show_discrete_stats=True,
    )

    # 2) Load configuration and create learners
    config = load_configuration()
    learners = create_learners(config)
    logger.debug(learners)

    # 3) Do transformation
    # data = transform(data)
    # dataset_report(
    #     data,
    #     preview_rows=10,
    #     show_all_features=True,
    #     show_all_metas=True,
    #     show_numeric_stats=True,
    #     show_discrete_stats=True,
    # )

    # 4) Split (here?)

    # 5) Preprocess
    preprocessor = PreprocessorList(
        preprocessors=(
            # Average/Most frequent
            Impute(method=Average()),
            # One-hot encoding/One feature per value
            Continuize(multinomial_treatment=Continuize.Indicators),
            # Standardization (z-score normalization)
            Normalize(norm_type=Normalize.NormalizeBySD),
        )
    )

    # TODO: somehow verify that the preprocessor is working correctly
    # (e.g. check for NaNs, check that features are continuous, etc.)
    # data = preprocessor(data)

    # 6) Evaluate
    # TODO: may be try to use a common wrapper for evaluation (check TestAndScore in experiment.py)
    evaluate(
        data,
        learners["logistic-regression"]
        + learners["random-forest"]
        + learners["tree"]
        + learners["gradient-boosting"]
        + learners["neural-network"]
        + learners["svm"],
        preprocessor,
    )
