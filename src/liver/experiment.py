from abc import ABC, abstractmethod

import json
import numpy as np
from sklearn.utils.multiclass import type_of_target
import pandas as pd
from Orange.data.pandas_compat import table_to_frame
from loguru import logger

from Orange.data import Domain, Table, DiscreteVariable

from Orange.preprocess import PreprocessorList, Impute, Average, Continuize, Normalize
from Orange.classification import (
    LogisticRegressionLearner,
    RandomForestLearner,
    TreeLearner,
    GBClassifier,
    NNClassificationLearner,
    SVMLearner,
)


from .load import load_dataset, load_configuration
from .utils import create_learners


class Experiment(ABC):
    _instances = {}

    def __new__(cls, *args, **kwargs):
        if cls is Experiment:
            raise TypeError(
                "Experiment is an abstract class and cannot be instantiated directly."
            )
        elif cls not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[cls] = instance

        return cls._instances[cls]

    def __init__(self):
        self._name = None
        self._data: dict[str, Table] | Table | None = None
        self._learners = None
        # Preprocessing is common for all experiments
        self._preprocessor = PreprocessorList(
            preprocessors=(
                # Average/Most frequent
                Impute(method=Average()),
                # One-hot encoding/One feature per value
                Continuize(multinomial_treatment=Continuize.Indicators),
                # Standardization (z-score normalization)
                Normalize(norm_type=Normalize.NormalizeBySD),
            )
        )

    def init(self, dataset: str | list[str]):
        """Load CSV file, configuration and create learners."""
        if isinstance(dataset, str):
            self._data = load_dataset(dataset)
        else:
            self._data = {ds: load_dataset(ds) for ds in dataset}
        config = load_configuration()
        self._learners = create_learners(config)
        logger.debug(self._learners)

    @abstractmethod
    def run(self):
        raise NotImplementedError("Subclasses must implement the run method.")

    @abstractmethod
    def plot(self):
        raise NotImplementedError("Subclasses must implement the plot method.")

    @property
    def data(self) -> dict[str, Table] | Table | None:
        return self._data

    @data.setter
    def data(self, value: Table):
        self._data = value


class TestAndScore:
    def __init__(self):
        pass

    def __call__(self, data, learners: list, preprocessor, metrics: dict):
        """TODO: add docstring."""
        # Split the data into train and test sets (80% train, 20% test, stratified, random state for reproducibility)
        train, test = sample(data, n=0.8, stratified=True, random_state=42)

        # Evaluate using TestOnTestData (train on train set, test on test set)
        scores = TestOnTestData(
            train_data=train,
            test_data=test,
            learners=learners,
            preprocessor=preprocessor,
            store_data=True,
        )
        # scores = CrossValidation(data, learners, k=5)
        logger.debug(scores)

        logger.info("CA:", CA(scores))
        logger.info("AUC:", AUC(scores))
        logger.info("F1:", F1(scores, average="weighted"))
        logger.info("Precision:", Precision(scores, average="weighted"))
        logger.info("Recall:", Recall(scores, average="weighted"))

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
