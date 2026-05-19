from loguru import logger
from pathlib import Path

from .utils import root

from Orange.data import Table

# TODO: change this path!
DATASETS_PATH: Path = root("datasets")
DATASETS: dict[str, str] = {
    'indian': DATASETS_PATH / "indian-liver-disease-dataset/Training_indian_liver_disease_dataset.csv",
    'hcv': DATASETS_PATH / "hcv-data/hcvdat0.csv",
    'liver': DATASETS_PATH / "liver-data/cleaned_data.csv",
}

def load_csv(dataset: str) -> Table:
    """Load CSV file for the specified dataset."""

    filename = DATASETS[dataset]
    logger.debug(filename)
    data = Table(str(filename))

    return data


def load_configuration():
    """TODO: write docstring."""
    import json

    with open(root("config.json"), "r") as fd:
        config = json.load(fd)

    logger.debug("Configuration:\n{}", json.dumps(config, indent=2))

    return config
