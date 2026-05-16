#!/usr/bin/env python3

# TODO: describe the workflow
"""Multi-class classification for Indian Liver Disease dataset.

Learners:

Preprocessing:

Evaluation:

"""

from pathlib import Path
import json, hashlib
from multiprocessing import Process
import numpy as np
from loguru import logger
from math import prod
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
# import lazy_loader as lazy

# Orange = lazy.load("Orange")

# Data
from Orange.data import Table, Domain

# Statistics
from Orange.statistics import distribution

# Preprocessing
from Orange.preprocess import Impute, Continuize, Normalize, PreprocessorList

# Learners
from Orange.classification import (
    LogisticRegressionLearner,
    RandomForestLearner,
    GBClassifier,
    NNClassificationLearner,
    TreeLearner,
    SVMLearner,
)

# Evaluation
from Orange.evaluation import (
    CrossValidation,
    TestOnTestData,
    CA,
    AUC,
    F1,
    Precision,
    Recall,
    MatthewsCorrCoefficient,
    testing,
)

def get_distribution(data: Table):
    # Distributions
    dist = distribution.get_distribution(data, data.domain["Liver_Disease_Type"])
    print(dist)


def sample():
    pass


def get_learners():
    from itertools import product

    # Learners grouped by family and variant hash
    learners_by_family: dict = {
        "logistic-regression": [],
        "random-forest": [],
        "svm": [],
        "decision-tree": [],
        "gradient-boosting": [],
        "neural-network": [],
    }

    config = json.load(open(Path(__file__).parent / "config.json"))

    LEARNER_MAP: dict = {
        "logistic-regression": LogisticRegressionLearner,
        "random-forest": RandomForestLearner,
        "svm": SVMLearner,
        "decision-tree": TreeLearner,
        "gradient-boosting": GBClassifier,
        "neural-network": NNClassificationLearner,
    }

    for name, params in config.items():
        cls = LEARNER_MAP[name]
        keys = params.keys()
        for combo in product(*params.values()):
            learners[name].append(cls(**dict(zip(keys, combo))))

        # Make sure that we have constructed the right amount of learners
        assert len(learners[name]) == prod(len(dim) for dim in params.values())

    return learners


def main():
    data = load()
    learners = get_learners()
    raise ValueError

    # ============
    # Data sampler: 80/20, stratified, replicable
    # ============
    train, test = testing.sample(data, n=0.8, stratified=True, random_state=42)
    logger.debug(f"Train set: {len(train)} rows")
    logger.debug(f"Test set: {len(test)} rows")

    # ============
    # Preprocessor
    #   default = mean for continuous, mode for discrete
    # ============
    imputer = Impute()
    continuizer = Continuize(
        multinomial_treatment=Continuize.Indicators  # TODO: verify that this is one-hot encoding
    )
    normalizer = Normalize(
        norm_type=Normalize.NormalizeBySD
    )  # z-score standardization: (x - mean) / std
    preprocessor = PreprocessorList([imputer, continuizer, normalizer])

    # Evaluate learners
    # learners =
    cv = CrossValidation(k=5, random_state=42)
    results = cv(
        data,
        learners,
        preprocessor=preprocessor,
    )

    # This adds predictions, probabilities, and fold index to a Table
    augmented = results.get_augmented_data(
        model_names=learners.keys(),
        include_attrs=True,
        include_predictions=True,
        include_probabilities=True,
    )

    # Save like any Orange Table
    augmented.save("cv_augmented_predictions.csv")

    # TODO: do this here or in Orange as final test?
    # results = TestOnTestData(train, test, learners, preprocessor)
