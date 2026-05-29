import json
from pathlib import Path

from loguru import logger
from Orange.data import Table

from .cli import AVAILABLE_CONFIGS
from .utils import root

DATASETS_PATH: Path = root("datasets")
DATASETS: dict[str, Path] = {
	"experiment1": DATASETS_PATH / "expr1" / "experiment1.tab",
	"experiment2": DATASETS_PATH / "expr2" / "experiment2.tab",
	"experiment3": DATASETS_PATH / "expr3" / "experiment3.tab",
}


def load_dataset(dataset: str) -> Table:
	"""Load file for the specified dataset.

	Args:
		dataset (str): Name of the dataset to load.
						Available options are: 'experiment1', 'experiment2', 'experiment3'.
	"""
	path = DATASETS[dataset]
	logger.info("Loading dataset from: {}", path)

	if not path.exists():
		raise FileNotFoundError(f"Dataset file does not exist: {path}")

	data = Table(str(path))

	logger.success(f"Dataset loaded successfully: {path}")
	logger.info("Loaded rows: {}", len(data))

	return data


def load_configuration(config: str = "default") -> dict:
	"""Load configuration (JSON) file."""

	if config not in AVAILABLE_CONFIGS:
		raise ValueError(
			f"Please, specify a valid configuration! Available are: {AVAILABLE_CONFIGS}"
		)

	configuration = root("configs", f"{config}.json")

	with open(configuration) as fd:
		data = json.load(fd)

	logger.success(f"Configuration loaded: {configuration}")

	return data
