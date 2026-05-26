from __future__ import annotations

import functools
from itertools import product
from pathlib import Path
import time

from loguru import logger
from Orange.classification import (
	GBClassifier,
	LogisticRegressionLearner,
	NNClassificationLearner,
	RandomForestLearner,
	SVMLearner,
	TreeLearner,
)


def root(*args):
	"""Get root folder of the project.
	TODO: maybe find a better way?

	Args:
		*args: path components to join with the project root.

	Returns:
		Path: the path to the project root joined with the provided path components.
	"""
	return Path(__file__).resolve().parents[2].joinpath(*args)


def create_learners(config):
	"""Create all combinations of learners by the specified configuration."""
	# Helper function to get all combinations of hyperparameters
	def get_combinations(params):
		return [dict(zip(params.keys(), combo, strict=True)) for combo in product(*params.values())]

	# 1) Logistic regression
	configuration = config["logistic-regression"]
	combos = get_combinations(configuration)
	logger.debug(f"LR combinations ({len(combos)}): {combos}")
	logistic_regressions = [LogisticRegressionLearner(**combo) for combo in combos]
	logger.debug(logistic_regressions)

	# 2) Random forest
	configuration = config["random-forest"]
	combos = get_combinations(configuration)
	logger.debug(f"RF combinations ({len(combos)}): {combos}")
	random_forests = [RandomForestLearner(**combo) for combo in combos]
	logger.debug(random_forests)

	# 3) Tree
	configuration = config["tree"]
	combos = get_combinations(configuration)
	logger.debug(f"Tree combinations ({len(combos)}): {combos}")
	trees = [TreeLearner(**combo) for combo in combos]
	logger.debug(trees)

	# 4) Gradient boosting
	configuration = config["gradient-boosting"]
	combos = get_combinations(configuration)
	logger.debug(f"GB combinations ({len(combos)}): {combos}")
	gradient_boostings = [GBClassifier(**combo) for combo in combos]
	logger.debug(gradient_boostings)

	# 5) Neural network
	configuration = config["neural-network"]
	combos = get_combinations(configuration)
	logger.debug(f"NN combinations ({len(combos)}): {combos}")
	neural_networks = [NNClassificationLearner(**combo) for combo in combos]
	logger.debug(neural_networks)

	# 6) SVM
	configuration = config["svm"]
	combos = []
	for subconfig in configuration:
		combos.extend(get_combinations(subconfig))
	logger.debug(f"SVM combinations ({len(combos)}): {combos}")
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
		logger.info(
			f"Function '{func.__name__}' executed in {end_time / 60:.2f} mins ({end_time:.4f} seconds)"
		)
		return result

	return wrapper
