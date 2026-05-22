from __future__ import annotations

from collections import Counter
import functools
from itertools import product
from pathlib import Path
import time
from typing import Any

from loguru import logger
import numpy as np
from Orange.classification import (
	GBClassifier,
	LogisticRegressionLearner,
	NNClassificationLearner,
	RandomForestLearner,
	SVMLearner,
	TreeLearner,
)
from Orange.data import Table, Variable


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
	"""TODO: add docstring."""

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


def get_variable_type(variable: Variable) -> str:
	"""
	Return a readable Orange variable type.
	"""
	if variable.is_discrete:
		return "discrete"
	if variable.is_continuous:
		return "continuous"
	if variable.is_string:
		return "string"
	if variable.is_time:
		return "time"

	return type(variable).__name__


def count_missing_values(array: Any) -> int:
	"""
	Count missing values safely.

	Orange usually stores X and Y as numeric numpy arrays,
	where missing values are represented with np.nan.
	"""
	if array is None:
		return 0

	if not hasattr(array, "size") or array.size == 0:
		return 0

	try:
		return int(np.isnan(array).sum())
	except TypeError:
		return 0


def log_separator(title: str | None = None) -> None:
	"""
	Log visual section separators.
	"""
	logger.info("=" * 80)

	if title:
		logger.info(title)
		logger.info("=" * 80)


def log_domain_summary(data: Table) -> None:
	"""
	Log general information about the Orange domain.
	"""
	domain = data.domain

	log_separator("ORANGE DATASET SUMMARY")

	logger.info("Rows:                {}", len(data))
	logger.info("Feature attributes:  {}", len(domain.attributes))
	logger.info("Class variables:     {}", len(domain.class_vars))
	logger.info("Meta attributes:     {}", len(domain.metas))

	logger.info("Has single class var: {}", domain.class_var is not None)

	if domain.class_var is not None:
		logger.info("Main class variable:  {}", domain.class_var.name)
	else:
		logger.warning("No main class variable is set.")


def log_target_info(data: Table) -> None:
	"""
	Log information about the target/class variable.

	For a normal classification task, Orange should have:
		- exactly one class variable
		- the class variable should be discrete
	"""
	domain = data.domain

	logger.info("")
	log_separator("TARGET VARIABLE INFO")

	if not domain.class_vars:
		logger.error("No target/class variable found.")
		logger.error("This dataset is not ready for classification.")
		logger.error("Check whether your .tab file marks the target column as class.")
		return

	if len(domain.class_vars) > 1:
		logger.warning(
			"Dataset has {} class variables. This may be a multi-target task.",
			len(domain.class_vars),
		)

	for class_var in domain.class_vars:
		target_type = get_variable_type(class_var)

		logger.info("Target name: {}", class_var.name)
		logger.info("Target type: {}", target_type)

		if class_var.is_discrete:
			logger.info("Target classes: {}", list(class_var.values))
			logger.success("Target is discrete, so it is suitable for classification.")
		else:
			logger.warning(
				"Target variable '{}' is not discrete. Orange may treat this as regression.",
				class_var.name,
			)


def log_feature_info(data: Table, show_all_features: bool = True) -> None:
	"""
	Log feature names and feature types.
	"""
	domain = data.domain

	logger.info("")
	log_separator("FEATURE INFO")

	if not domain.attributes:
		logger.warning("No feature attributes found.")
		return

	type_counts = Counter(get_variable_type(attr) for attr in domain.attributes)

	logger.info("Feature type summary:")

	for feature_type, count in type_counts.items():
		logger.info("  {}: {}", feature_type, count)

	if not show_all_features:
		return

	logger.info("")
	logger.info("Feature list:")

	for index, attr in enumerate(domain.attributes, start=1):
		attr_type = get_variable_type(attr)

		if attr.is_discrete:
			logger.info(
				"{}. {} | {} | values={}",
				index,
				attr.name,
				attr_type,
				list(attr.values),
			)
		else:
			logger.info(
				"{}. {} | {}",
				index,
				attr.name,
				attr_type,
			)


def log_meta_info(data: Table, show_all_metas: bool = True) -> None:
	"""
	Log meta attributes.

	Metas are columns that are not used as model features by default.
	Think of them as descriptive columns, IDs, names, comments, etc.
	"""
	domain = data.domain

	logger.info("")
	log_separator("META ATTRIBUTE INFO")

	if not domain.metas:
		logger.info("No meta attributes.")
		return

	logger.info("Meta attributes: {}", len(domain.metas))

	if not show_all_metas:
		return

	for index, meta in enumerate(domain.metas, start=1):
		logger.info(
			"{}. {} | {}",
			index,
			meta.name,
			get_variable_type(meta),
		)


def log_class_distribution(data: Table) -> None:
	"""
	Log class distribution for single-target classification.

	This is very important for medical datasets because it shows imbalance.
	"""
	domain = data.domain

	logger.info("")
	log_separator("CLASS DISTRIBUTION")

	if len(domain.class_vars) != 1:
		logger.warning("Class distribution is shown only for single-target classification.")
		return

	class_var = domain.class_vars[0]

	if not class_var.is_discrete:
		logger.warning("Class distribution is available only for discrete target variables.")
		return

	y = np.asarray(data.Y).ravel()

	if y.size == 0:
		logger.warning("Target array Y is empty.")
		return

	known_y = y[~np.isnan(y)]
	missing_count = len(y) - len(known_y)

	if len(known_y) == 0:
		logger.error("All target values are missing.")
		return

	counts = Counter(int(value) for value in known_y)
	total = len(known_y)

	logger.info("Total known target values: {}", total)

	for class_index, class_name in enumerate(class_var.values):
		count = counts.get(class_index, 0)
		percent = count / total * 100

		logger.info(
			"{}: {} rows ({:.2f}%)",
			class_name,
			count,
			percent,
		)

	if missing_count > 0:
		logger.warning("Missing target values: {}", missing_count)

	if len(class_var.values) == 2:
		logger.success("Binary classification target detected.")
	else:
		logger.info("Multiclass classification target detected.")


def log_missing_values(data: Table) -> None:
	"""
	Log missing values in X, Y and metas.
	"""
	logger.info("")
	log_separator("MISSING VALUES")

	missing_x = count_missing_values(data.X)
	missing_y = count_missing_values(data.Y)
	missing_metas = count_missing_values(data.metas)

	logger.info("Missing feature values X: {}", missing_x)
	logger.info("Missing target values Y:  {}", missing_y)
	logger.info("Missing meta values:      {}", missing_metas)

	total_x_values = data.X.size if hasattr(data.X, "size") else 0

	if total_x_values > 0:
		missing_x_percent = missing_x / total_x_values * 100
		logger.info("Missing X percentage:    {:.2f}%", missing_x_percent)


def log_numeric_feature_statistics(data: Table) -> None:
	"""
	Log basic statistics for continuous features.

	This is useful before scaling, SMOTE, distance-based models, etc.
	"""
	domain = data.domain

	logger.info("")
	log_separator("NUMERIC FEATURE STATISTICS")

	continuous_features = [
		(index, attr) for index, attr in enumerate(domain.attributes) if attr.is_continuous
	]

	if not continuous_features:
		logger.info("No continuous features found.")
		return

	for column_index, attr in continuous_features:
		column = data.X[:, column_index]

		if column.size == 0:
			continue

		known_values = column[~np.isnan(column)]

		if known_values.size == 0:
			logger.warning("{}: all values are missing", attr.name)
			continue

		logger.info(
			"{} | min={:.4f}, max={:.4f}, mean={:.4f}, std={:.4f}",
			attr.name,
			float(np.min(known_values)),
			float(np.max(known_values)),
			float(np.mean(known_values)),
			float(np.std(known_values)),
		)


def log_discrete_feature_statistics(data: Table) -> None:
	"""
	Log value counts for discrete features.
	"""
	domain = data.domain

	logger.info("")
	log_separator("DISCRETE FEATURE STATISTICS")

	discrete_features = [
		(index, attr) for index, attr in enumerate(domain.attributes) if attr.is_discrete
	]

	if not discrete_features:
		logger.info("No discrete features found.")
		return

	for column_index, attr in discrete_features:
		column = data.X[:, column_index]
		known_values = column[~np.isnan(column)]

		logger.info("Feature: {}", attr.name)

		if known_values.size == 0:
			logger.warning("  All values are missing.")
			continue

		counts = Counter(int(value) for value in known_values)
		total = len(known_values)

		for value_index, value_name in enumerate(attr.values):
			count = counts.get(value_index, 0)
			percent = count / total * 100

			logger.info(
				"  {}: {} rows ({:.2f}%)",
				value_name,
				count,
				percent,
			)


def log_data_preview(data: Table, rows: int = 10) -> None:
	"""
	Log the first N rows.

	Orange rows display values according to the domain,
	so this is usually more readable than raw data.X.
	"""
	logger.info("")
	log_separator("DATA PREVIEW")

	rows_to_show = min(rows, len(data))

	if rows_to_show == 0:
		logger.warning("Dataset is empty.")
		return

	for index, row in enumerate(data[:rows_to_show], start=1):
		logger.info("{}: {}", index, row)


def dataset_report(
	data: Table,
	*,
	preview_rows: int = 10,
	show_all_features: bool = True,
	show_all_metas: bool = True,
	show_numeric_stats: bool = True,
	show_discrete_stats: bool = True,
) -> None:
	"""
	Full inspection function for an Orange classification dataset.
	"""
	log_domain_summary(data)
	log_target_info(data)
	log_feature_info(data, show_all_features=show_all_features)
	log_meta_info(data, show_all_metas=show_all_metas)
	log_class_distribution(data)
	log_missing_values(data)

	if show_numeric_stats:
		log_numeric_feature_statistics(data)

	if show_discrete_stats:
		log_discrete_feature_statistics(data)

	log_data_preview(data, rows=preview_rows)

	logger.info("")
	logger.success("Dataset inspection finished.")
