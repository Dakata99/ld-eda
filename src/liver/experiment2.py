"""
TODO: add docstring.
Binary classification for indian dataset.
"""
import json
import numpy as np
from sklearn.utils.multiclass import type_of_target
import pandas as pd
from Orange.data.pandas_compat import table_to_frame
from loguru import logger

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
from .utils import create_learners, profiler

class EditDomain:
    def __init__(self):
        pass

    def remap(self,
              data: Table,
              attribute: str,
              mapping: dict[str, str],
              new_attr_name: str | None = None
              ):
        """
        Remap values of a discrete attribute/feature.

        Example:
            mapping = {
                "1": "Male",
                "2": "Female",
            }
        """
        old_attrs = list(data.domain.attributes)
        old_var = data.domain[attribute]

        if old_var not in old_attrs:
            raise ValueError(
                f"'{attribute}' is not an attribute. "
                "This function remaps feature columns, not class/metas."
            )

        if not old_var.is_discrete:
            raise TypeError(f"'{attribute}' is not discrete.")

        col_index = old_attrs.index(old_var)

        old_values = list(old_var.values)
        logger.debug(f"Old values for '{attribute}': {old_values}")

        new_labels = []
        for label in old_values:
            new_label = mapping.get(label, label)
            if new_label not in new_labels:
                new_labels.append(new_label)

        new_var = DiscreteVariable(
            new_attr_name or old_var.name,
            values=new_labels,
        )

        X_new = data.X.copy()
        old_col = X_new[:, col_index].copy()
        new_col = np.full(len(data), np.nan, dtype=float)

        label_to_new_index = {
            label: i for i, label in enumerate(new_labels)
        }

        for old_index, old_label in enumerate(old_values):
            mapped_label = mapping.get(old_label, old_label)
            mapped_index = label_to_new_index[mapped_label]

            new_col[old_col == old_index] = mapped_index

        X_new[:, col_index] = new_col

        new_attrs = old_attrs.copy()
        new_attrs[col_index] = new_var

        new_domain = Domain(
            attributes=new_attrs,
            class_vars=data.domain.class_var,
            metas=data.domain.metas,
        )

        return Table.from_numpy(
            domain=new_domain,
            X=X_new,
            Y=data.Y.copy(),
            metas=data.metas.copy(),
            W=data.W.copy() if data.has_weights() else None,
            ids=data.ids.copy() if hasattr(data, "ids") else None,
        )


def transform(data):
    """TODO: add docstring."""
    target = data.domain["Liver_Disease_Type"]

    assert isinstance(target, DiscreteVariable), "Target variable must be discrete!"
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
    evaluator = TestOnTestData(store_data=False) # set store_data to True if we want to keep the augmented data with predictions, probabilities, etc.
    # CrossValidation(store_data=False, k=5)
    scores = evaluator(
        train,
        test,
        learners,
        preprocessor=preprocessor,
        callback=progress_callback,
    )
    logger.debug(scores)

    # logger.error("Actual shape: {}", scores.actual.shape)
    # logger.error("Actual type_of_target: {}", type_of_target(scores.actual))
    # logger.error("Actual unique: {}", np.unique(scores.actual))

    # logger.error("Predicted full shape: {}", scores.predicted.shape)

    # for i, learner in enumerate(learners):
    #     y_pred = scores.predicted[i]

    #     logger.error("=" * 80)
    #     logger.error("Learner: {}", repr(learner))
    #     logger.error("Prediction shape: {}", y_pred.shape)
    #     logger.error("Prediction dtype: {}", y_pred.dtype)
    #     logger.error("Prediction type_of_target: {}", type_of_target(y_pred))
    #     logger.error("Prediction first 20: {}", y_pred[:20])
    #     logger.error("Prediction unique first 30: {}", np.unique(y_pred)[:30])

    #     is_integer_like = np.all(np.isclose(y_pred, np.rint(y_pred)))
    #     logger.error("Prediction is integer-like: {}", is_integer_like)

    logger.debug(f"CA: {CA(scores)}")
    logger.debug(f"AUC: {AUC(scores)}")
    logger.debug(f"F1: {F1(scores)}")
    logger.debug(f"Precision: {Precision(scores)}")
    logger.debug(f"Recall: {Recall(scores)}")

    metrics = {
        "CA": CA,
        "AUC": AUC,
        "Precision": Precision,
        "Recall": Recall,
        "F1": F1,
        "MCC": MatthewsCorrCoefficient,
    }

    # Write results into a CSV file
    rows = []

    # Loop through learners
    for i, learner in enumerate(learners):
        row = {
            "Learner": repr(learner)
        }

        for name, metric in metrics.items():
            values = metric(scores)
            row[name] = values[i]

        rows.append(row)

    df = pd.DataFrame(rows)
    logger.success(df.to_string(index=False))
    df.to_csv("evaluation_results.csv", index=False)


def main():
    """TODO: write docstring."""
    logger.info('Running experiment 2: binary classification for indian dataset')

    # 1) Load the dataset (CSV file)
    data = load_csv('indian')

    # 2) Load configuration and create learners
    config = load_configuration()
    learners = create_learners(config)

    logger.debug(learners)

    # 3) Do transformation
    # TODO: INSTEAD OF DOING THIS, JUST EXPORT THE DATA FROM ORANGE WITH THE MAPPED VALUES.
    # THIS IS A LOT OF WORK AND CAN BE ERROR-PRONE.
    # ALSO, THIS IS NOT REALLY THE FOCUS OF THE EXPERIMENT, SO IT'S BETTER TO JUST PREPARE THE DATA IN ORANGE AND THEN LOAD IT HERE.
    # TODO: export the data in .tab or .pkl file, so we can load it here and directly know which is the target variable, which are the features, etc. without having to do all this transformation and remapping here.
    ed = EditDomain()
    data = ed.remap(
        data=data,
        attribute="Liver_Disease_Type",
        mapping={
            'Normal': 'Healthy',
            'Alcoholic_Liver_Disease': 'Sick',
            'Cirrhosis': 'Sick',
            'Fatty_Liver': 'Sick',
            'Hepatitis_B': 'Sick',
            'Hepatitis_C': 'Sick',
        }
    )
    data = transform(data)

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
        learners["logistic-regression"] +
        learners["random-forest"] +
        learners["tree"] +
        learners["gradient-boosting"] +
        learners["neural-network"] +
        learners["svm"],
        preprocessor
    )
