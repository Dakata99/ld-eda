from functools import partial
from itertools import chain
from pathlib import Path

from loguru import logger
from Orange.data import Table
from Orange.evaluation import AUC, CA, F1, MatthewsCorrCoefficient, Precision, Recall
from Orange.evaluation.testing import TestOnTestData, CrossValidation
from Orange.preprocess import Average, Continuize, Impute, Normalize, PreprocessorList
import pandas as pd

from .load import load_configuration, load_dataset
from .utils import create_learners, dataset_report, profiler, root

from Orange.data import Table

OUTPUT_DIR: Path = root("results")
CSV_FILE: str = "experiment{experiment}-{config}-{method}.csv"


class TestAndScore:
	def __init__(self, data, learners: list):
		self._data = data
		self._learners = learners
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
	def train(self, exprid: int, method: str):
		"""Method for training learnears."""
		# Load training data
		train = Table(str(root("datasets", f"expr{exprid}", f"expr{exprid}-train-data.tab")))

		# Evaluate models with the chosen method
		if method == 'cv':
			logger.info(f"CrossValidation: evaluating {len(self._learners)} learners...")
			def progress_callback(progress: float) -> None:
				logger.info("Progress: {:.2f}%", round(progress * 100))

			cv = CrossValidation()
			self._scores = cv(
				train,
				self._learners,
				preprocessor=self._preprocessor,
				callback=progress_callback,
			)
		else:
			logger.info(f"TestOnTestData: evaluating {len(self._learners)} learners...")
			# Evaluate using TestOnTestData (train on train set, test on test set)
			def progress_callback(progress: float) -> None:
				done = round(progress * len(self._learners))
				logger.info(
					"Finished {}/{} learners ({:.2f}%)",
					done,
					len(self._learners),
					progress * 100,
				)

			test = Table(str(root("datasets", f"expr{exprid}", f"expr{exprid}-test-data.tab")))

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
		"""Metrics evalution."""
		sick_index = healthy_index = None
		if exprid in [2, 3]:
			sick_index = list(self._scores.domain.class_var.values).index("Sick")
			healthy_index = list(self._scores.domain.class_var.values).index("Healthy")

		# NOTE: priority of the metrics is preserved from here!
		# Will be ordered in the CSV file as here and plotting will keep this priority!
		metrics: dict[int, dict] = {
			1: {
				"Recall(weighted)": partial(Recall, target=None, average="weighted"),
				"F1(weighted)": partial(F1, target=None, average="weighted"),
				"MCC": MatthewsCorrCoefficient,
				"Precision(weighted)": partial(Precision, target=None, average="weighted"),
				"AUC": AUC,
				"CA": CA,
			},
			2: {
				"Recall(Sick)": partial(Recall, target=sick_index),
				"Recall(weighted)": partial(Recall, average="weighted"),
				"F1(Sick)": partial(F1, target=sick_index),
				"F1(weighted)": partial(F1, average="weighted"),
				"MCC": MatthewsCorrCoefficient,
				"Precision(Sick)": partial(Precision, target=sick_index),
				"AUC": AUC,
				"CA": CA,
			},
			3: {
				"Recall(Sick)": partial(Recall, target=sick_index),
				"Recall(weighted)": partial(Recall, average="weighted"),
				"F1(Sick)": partial(F1, target=sick_index),
				"F1(weighted)": partial(F1, average="weighted"),
				"MCC": MatthewsCorrCoefficient,
				"Precision(Sick)": partial(Precision, target=sick_index),
				"AUC": AUC,
				"CA": CA,
			},
		}

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


def main(exprid: int, method: str, learners_group: list, configuration: str = "default") -> None:
	"""Main function for evaluation an experiment."""
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
	ts.train(exprid, method)
	ts.eval(exprid, CSV_FILE.format(experiment=exprid, config=configuration, method=method))
