"""
TODO: add docstring.
Binary classification for all 3 dataset.
"""

from loguru import logger
from itertools import chain
import pandas as pd

from .load import load_dataset, load_configuration
from .utils import create_learners, dataset_report, profiler

from Orange.preprocess import PreprocessorList, Impute, Average, Continuize, Normalize

from Orange.evaluation.testing import CrossValidation, TestOnTestData, sample
from Orange.evaluation import Recall, F1, Precision, CA, AUC, MatthewsCorrCoefficient


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
	)  # type: ignore
	logger.debug(scores)

	# TODO: decide which average to use for binary classification.
	metrics = {
		"CA": CA,
		"AUC": AUC,
		"Precision": Precision,
		"Recall": Recall,
		"F1": F1,
		"MCC": MatthewsCorrCoefficient,
	}
	for name, metric in metrics.items():
		values = metric(scores)
		logger.debug(f"{name}: {values}")

	# Write results into a CSV file
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
	df.to_csv("experiment3-evaluation-results.csv", index=False)


def main(learner_group: str = "all", configuration: str = "global") -> None:
	"""TODO: write docstring."""
	logger.info("Running experiment 3: binary classification for all 3 datasets")

	# 1) Load the dataset (CSV file)
	data = load_dataset("experiment3")
	dataset_report(
		data,
		preview_rows=10,
		show_all_features=True,
		show_all_metas=True,
		show_numeric_stats=True,
		show_discrete_stats=True,
	)

	# 2) Load configuration and create learners
	config = load_configuration(configuration)
	learners = create_learners(config)
	logger.debug(learners)

	# 3) Preprocess
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

	# 4) Evaluate
	if learner_group == "all":
		learners_to_evaluate = list(chain.from_iterable(learners.values()))
	else:
		learners_to_evaluate = learners[learner_group]

	evaluate(
		data,
		learners_to_evaluate,
		preprocessor,
	)
