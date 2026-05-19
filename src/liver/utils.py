from pathlib import Path
from loguru import logger
from itertools import product
import time
import functools

from Orange.classification import (
    LogisticRegressionLearner,
    RandomForestLearner,
    TreeLearner,
    GBClassifier,
    NNClassificationLearner,
    SVMLearner,
)

def root(*args):
    """TODO: write docstring.
    TODO: maybe find a better way? --- IGNORE ---

    Args:
        *args: path components to join with the project root.

    Returns:
        Path: the path to the project root joined with the provided path components.
    """
    # TODO: maybe find a better way?
    return Path(__file__).resolve().parents[2].joinpath(*args)


def create_learners(config):
    """TODO: add docstring."""

    # Helper function to get all combinations of hyperparameters
    get_combinations = lambda params: [dict(zip(params.keys(), combo)) for combo in product(*params.values())]

    # 1) Logistic regression
    configuration = config["logistic-regression"]
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

def profiler(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter() - start_time
        logger.info(f"Function '{func.__name__}' executed in {end_time / 60:.2f} mins ({end_time:.4f} seconds)")
        return result
    return wrapper
