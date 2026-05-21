from functools import partial
from itertools import chain
from pathlib import Path
import pandas as pd
from loguru import logger

from Orange.preprocess import PreprocessorList, Impute, Average, Continuize, Normalize
from Orange.evaluation.testing import TestOnTestData, sample
from Orange.evaluation import Recall, F1, Precision, CA, AUC, MatthewsCorrCoefficient

from .load import load_dataset, load_configuration
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

	def eval(self, metrics: dict, output_filename: str):
		"""TODO: add docstring."""
		for name, metric in metrics.items():
			values = metric(self._scores)
			logger.debug(f"{name}: {values}")

		# Write results into a CSV file
		rows = []
		# Loop through learners
		for i, learner in enumerate(self._learners):
			row = {"Learner": repr(learner)}

			for name, metric in metrics.items():
				values = metric(self._scores)
				row[name] = values[i]

			rows.append(row)

		df = pd.DataFrame(rows)
		logger.success(df.to_string(index=False))

		if not OUTPUT_DIR.exists():
			OUTPUT_DIR.mkdir(parents=True)
		df.to_csv(OUTPUT_DIR / output_filename, index=False)


# TODO: decide which average to use for multiclass classification (e.g. weighted, macro, micro)
METRICS: dict[int, dict] = {
	1: {
		"CA": CA,
		"AUC": AUC,
		"Precision(average=macro)": partial(Precision, average="macro"),
		"Recall(average=macro)": partial(Recall, average="macro"),
		"F1(average=macro)": partial(F1, average="macro"),
		"MCC": MatthewsCorrCoefficient,
	},
	2: {
		"CA": CA,
		"AUC": AUC,
		"Precision": Precision,
		"Recall": Recall,
		"F1": F1,
		"MCC": MatthewsCorrCoefficient,
	},
	3: {
		"CA": CA,
		"AUC": AUC,
		"Precision": Precision,
		"Recall": Recall,
		"F1": F1,
		"MCC": MatthewsCorrCoefficient,
	},
}


def main(exprid: int, learner_group: str = "all", configuration: str = "global") -> None:
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

	# 4) Evaluate
	if learner_group == "all":
		learners_to_evaluate = list(chain.from_iterable(learners.values()))
	else:
		learners_to_evaluate = learners[learner_group]

	ts = TestAndScore(data, learners_to_evaluate)
	ts.train()
	ts.eval(METRICS[exprid], f"new-experiment{exprid}-evaluation-results.csv")
