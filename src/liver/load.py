from loguru import logger
from pathlib import Path

# TODO: change this path!
DATASETS_PATH: Path = Path(f'/mnt/c/Users/Daniel/My Drive (192knz@unibit.bg)/4 year (2025-2026)/Summer semmeseter/Diploma/datasets/')
DATASETS: dict[str, str] = {
    'indian': DATASETS_PATH / "indian-liver-disease-dataset/Training_indian_liver_disease_dataset.csv",
    'hcv': DATASETS_PATH / "hcv-data/hcvdat0.csv",
    'liver': DATASETS_PATH / "liver-data/cleaned_data.csv",
}

def load_csv(dataset: str):
    """Load CSV file for the specified dataset."""
    from Orange.data import Table

    filename = DATASETS[dataset]
    logger.debug(filename)
    data = Table(str(filename))

    return data


def load_configuration():
    """TODO: write docstring."""
    import json

    # TODO: maybe find a better way?
    project_root = Path(__file__).resolve().parents[2]
    logger.debug(project_root)

    with open(project_root / "config.json") as fd:
        config = json.load(fd)

    logger.debug("Configuration:\n{}", json.dumps(config, indent=2))

    return config
