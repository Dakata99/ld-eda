from functools import partial
from itertools import chain
from pathlib import Path

from loguru import logger
from Orange.evaluation import AUC, CA, F1, MatthewsCorrCoefficient, Precision, Recall
from Orange.evaluation.testing import TestOnTestData, sample
from Orange.preprocess import Average, Continuize, Impute, Normalize, PreprocessorList
import pandas as pd

from .load import load_configuration, load_dataset
from .utils import create_learners, dataset_report, profiler, root

OUTPUT_DIR: Path = root("results")


class TestAndScore:
	def __init__(self, data, learners: list):
		self._data = data
		self._learners = learners
		# TODO: somehow verify that the preprocessor is working correctly
		# (e.g. check for NaNs, check that features are continuous, etc.)
		# data = preprocessor(data)
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
		self._scores = None

	@profiler
	def train(self):
		"""TODO: add docstring."""
		# Split the data into train and test sets (80% train, 20% test, stratified, random state for reproducibility)
		train, test = sample(self._data, n=0.8, stratified=True, random_state=42)

		def progress_callback(progress: float) -> None:
			done = round(progress * len(self._learners))
			logger.info(
				"Finished {}/{} learners ({:.2f}%)",
				done,
				len(self._learners),
				progress * 100,
			)

		logger.info(f"Evaluating {len(self._learners)} learners...")

		# Evaluate using TestOnTestData (train on train set, test on test set)
		# Set store_data to True if we want to keep the augmented data with predictions, probabilities, etc.
		evaluator = TestOnTestData(store_data=False)
		self._scores = evaluator(
			train,
			test,
			self._learners,
			preprocessor=self._preprocessor,
			callback=progress_callback,
		)  # type: ignore

	def eval(self, exprid: int, output_filename: str):
		"""TODO: add docstring."""
		sick_index = healthy_index = None
		if exprid in [2, 3]:
			sick_index = list(self._scores.domain.class_var.values).index("Sick")
			healthy_index = list(self._scores.domain.class_var.values).index("Healthy")

		# TODO: decide which average to use for multiclass classification (e.g. weighted, macro, micro)
		# NOTE: priority of the metrics is preserved from here!
		# Will be ordered in the CSV file as here and plotting will keep this priority!
		metrics: dict[int, dict] = {
			1: {
				"Recall(macro)": partial(Recall, average="macro"),
				"F1(macro)": partial(F1, average="macro"),
				"MCC": MatthewsCorrCoefficient,
				"Precision(macro)": partial(Precision, average="macro"),
				"AUC": AUC,
				"CA": CA,
			},
			# TODO: decide which to use for binary classification
			2: {
				"Recall(Sick)": partial(Recall, target=sick_index),
				"Recall(macro)": partial(Recall, average="macro"),
				"F1(Sick)": partial(F1, target=sick_index),
				"F1(macro)": partial(F1, average="macro"),
				"MCC": MatthewsCorrCoefficient,
				"Precision(Sick)": partial(Precision, target=sick_index),
				"AUC": AUC,
				"CA": CA,
			},
			3: {
				"Recall(Sick)": partial(Recall, target=sick_index),
				"F1(Sick)": partial(F1, target=sick_index),
				"MCC": MatthewsCorrCoefficient,
				"Precision(Sick)": partial(Precision, target=sick_index),
				"AUC": AUC,
				"CA": CA,
			},
		}

		for name, metric in metrics[exprid].items():
			values = metric(self._scores)
			logger.debug(f"{name}: {values}")

		# Write results into a CSV file
		rows = []
		# Loop through learners
		for i, learner in enumerate(self._learners):
			row = {"Learner": repr(learner)}

			for name, metric in metrics[exprid].items():
				values = metric(self._scores)
				row[name] = values[i]

			rows.append(row)

		df = pd.DataFrame(rows)
		logger.success(df.to_string(index=False))

		if not OUTPUT_DIR.exists():
			OUTPUT_DIR.mkdir(parents=True)
		df.to_csv(OUTPUT_DIR / output_filename, index=False)


def main(exprid: int, learners_group: list, configuration: str = "default") -> None:
	"""TODO: write docstring."""
	logger.info(f"Running experiment {exprid}")

	# 1) Load the dataset (CSV file)
	data = load_dataset(f"experiment{exprid}")
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

	# 3) Evaluate
	learners_to_evaluate = []
	if learners_group is None:
		learners_to_evaluate = list(chain.from_iterable(learners.values()))
	else:
		for group in learners_group:
			if group in learners:
				learners_to_evaluate.extend(learners[group])

	ts = TestAndScore(data, learners_to_evaluate)
	ts.train()
	ts.eval(exprid, f"experiment{exprid}-{configuration}-evaluation-results.csv")
