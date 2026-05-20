import json
from loguru import logger
from pathlib import Path

from .utils import root

from Orange.data import Table

# TODO: change this path!
DATASETS_PATH: Path = root("datasets")
DATASETS: dict[str, Path] = {
    # "indian": DATASETS_PATH / "indian-liver-disease-dataset/Training_indian_liver_disease_dataset.csv",
    # "hcv": DATASETS_PATH / "hcv-data/hcvdat0.csv",
    # "liver": DATASETS_PATH / "liver-data/cleaned_data.csv",
    "experiment1": DATASETS_PATH / "experiment1.tab",
    "experiment2": DATASETS_PATH / "experiment2.tab",
    "experiment3": DATASETS_PATH / "experiment3.tab",
}


def load_dataset(dataset: str) -> Table:
    """Load file for the specified dataset.

    Args:
        dataset (str): Name of the dataset to load.
                       Available options are: 'experiment1', 'experiment2', 'experiment3'.
    """
    filename = DATASETS[dataset]
    logger.debug(f'Loaded file: {filename}')
    data = Table(str(filename))

    return data


def load_configuration(config: str = "global") -> dict:
    """TODO: write docstring."""

    available_configs = ('global', 'experiment1', 'experiment2', 'experiment3')
    if config not in available_configs:
        raise ValueError(f'Please, specify a valid configuration! Available are: {available_configs}')

    configuration = root("configs", f"{config}.json")

    with open(configuration, "r") as fd:
        data = json.load(fd)

    logger.success(f"Configuration loaded: {configuration}")

    return data
