"""
TODO: add docstring.
Multiclass classification for indian dataset.
"""
import json
import numpy as np
import pandas as pd
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

    # Rebuild Table with proper integer encoding (critical for Orange metrics)
    # new_domain = Domain(data.domain.attributes, data.domain.class_var, data.domain.metas)
    # data = Table(domain, data)

    info = {
        "target": str(data.domain.class_var),
        "labels": list(data.domain.class_var.values),
        "rows": len(data),
        "features": len(data.domain.attributes),
        "attributes": [str(a) for a in data.domain.attributes],
    }

    logger.debug("Dataset info:\n{}", json.dumps(info, indent=2))

    logger.error(f"Class variable: {data.domain.class_var}")
    logger.error(f"Class variable type: {type(data.domain.class_var)}")
    logger.error(f"Is discrete: {data.domain.class_var.is_discrete}")
    logger.error(f"Is continuous: {data.domain.class_var.is_continuous}")
    logger.error(f"Y dtype: {data.Y.dtype}")
    logger.error(f"Unique Y values: {np.unique(data.Y)[:20]}")

    return data

def get_combinations(params):
    """TODO: add docstring."""
    keys = list(params.keys())
    values = list(params.values())

    return [dict(zip(keys, combo)) for combo in product(*values)]

def create_learners(config):
    """TODO: add docstring."""
    # 1) Logistic regression
    logistic_regression = config["logistic-regression"]
    logger.info(logistic_regression)

    combos = get_combinations(logistic_regression)
    logger.debug(f"Logistic regression combinations: {len(combos)}")
    for combo in combos:
        combo["class_weight"] = 'balanced'
        logger.debug(combo)

    lrs = [LogisticRegressionLearner(**combo) for combo in combos]
    logger.debug(lrs)

    # 2) Random forest
    random_forest = config["random-forest"]
    combos = get_combinations(random_forest)
    logger.debug(f"Random forest combinations: {len(combos)}")
    for combo in combos:
        combo["class_weight"] = True
        logger.debug(combo)
    rfs = [RandomForestLearner(**combo) for combo in combos]
    logger.debug(rfs)

    # 3) Tree
    tree = config["tree"]
    combos = get_combinations(tree)
    logger.debug(f"Tree combinations: {len(combos)}")
    for combo in combos:
        logger.debug(combo)
    trees = [TreeLearner(**combo) for combo in combos]
    logger.debug(trees)

    # # 4) Gradient boosting
    # gradient_boosting = config["gradient-boosting"]
    # combos = get_combinations(gradient_boosting)
    # logger.debug(f"Gradient boosting combinations: {len(combos)}")
    # for combo in combos:
    #     logger.debug(combo)

    # # 5) Neural network
    # neural_network = config["neural-network"]
    # combos = get_combinations(neural_network)
    # logger.debug(f"Neural network combinations: {len(combos)}")
    # for combo in combos:
    #     logger.debug(combo)

    # 6) SVM
    # FIXME: kerner issues
    # svm = config["svm"]
    # combos = get_combinations(svm)
    # logger.debug(f"SVM combinations: {len(combos)}")
    # for combo in combos:
    #     logger.debug(combo)

    return {
        "logistic-regression": lrs,
        "random-forest": rfs,
        "tree": trees,
        # "gradient-boosting": gradient_boosting,
        # "neural-network": neural_network,
        # "svm": svm,
    }


def evaluate(data, learners: list, preprocessor):
    train, test = sample(data, n=0.8, stratified=True, random_state=42)
    scores = TestOnTestData(
        train_data=train,
        test_data=test,
        learners=learners,
        preprocessor=preprocessor
    )
    # scores = CrossValidation(data, learners["logistic-regression"], k=5)
    logger.debug(scores)

    import numpy as np
    from sklearn.utils.multiclass import type_of_target

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
        # "AUC": AUC,
        # "Precision_weighted": lambda res: Precision(scores, average="weighted"),
        # "Recall_weighted": lambda res: Recall(scores, average="weighted"),
        # "F1_weighted": lambda res: F1(scores, average="weighted"),
        # "MCC": MatthewsCorrCoefficient,
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

    # # Use sklearn metrics with proper data extraction
    # import numpy as np
    # from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
    
    # y_test = scores.actual.ravel()
    
    # results = []
    # for i, learner in enumerate(learners["logistic-regression"]):
    #     # Extract class predictions from probabilities (argmax)
    #     y_pred = np.argmax(scores.probabilities[i], axis=1)
        
    #     # Ensure both are same type
    #     y_test_int = y_test.astype(int)
    #     y_pred_int = y_pred.astype(int)
        
    #     recall = recall_score(y_test_int, y_pred_int, average='macro', zero_division=0)
    #     precision = precision_score(y_test_int, y_pred_int, average='macro', zero_division=0)
    #     f1 = f1_score(y_test_int, y_pred_int, average='macro', zero_division=0)
    #     accuracy = accuracy_score(y_test_int, y_pred_int)
        
    #     results.append({
    #         'learner': str(learner),
    #         'accuracy': accuracy,
    #         'recall': recall,
    #         'precision': precision,
    #         'f1': f1
    #     })
    #     logger.info(f"Learner {i}: Accuracy={accuracy:.4f}, Recall={recall:.4f}, Precision={precision:.4f}, F1={f1:.4f}")
    
    # logger.debug(f"All results: {results}")


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

    # 3)
    config = load_configuration()
    learners = create_learners(config)

    logger.debug(learners)

    # 4) Split

    # 5) Preprocess
    preprocessor = PreprocessorList(preprocessors=(
            Impute(method=Average()), # Average/Most frequent
            Continuize(multinomial_treatment=Continuize.Indicators), # One-hot encoding/One feature per value
            Normalize(norm_type=Normalize.NormalizeBySD) # Standardization (z-score normalization)
        )
    )

    # TODO: somehow verify that the preprocessor is working correctly (e.g. check for NaNs, check that features are continuous, etc.)
    # data = preprocessor(data)

    # 6) Evaluate
    evaluate(data, learners["logistic-regression"], preprocessor)
    # evaluate(
    #     data,
    #     [
    #         LogisticRegressionLearner(C=0.1),
    #         LogisticRegressionLearner(penalty='l1', C=0.1)
    #     ],
    #     preprocessor
    # )
