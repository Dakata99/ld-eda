"""
TODO: add docstring.
Binary classification for indian dataset.
"""
from loguru import logger

from .load import load_csv

from Orange.preprocess import (
    PreprocessorList,
    Impute,
    Average,
    Continuize,
    Normalize
)

def main():
    """TODO: write docstring."""
    logger.info('Running experiment 2: binary classification for indian dataset')

    # 1) Load the dataset (CSV file)
    data = load_csv('indian')

    # 2) Do transformation

    # 3) Load configuration and create learners

    # 4) Split

    # 5) Preprocess
    preprocessor = PreprocessorList(
        preprocessors=(
            # Average/Most frequent
            Impute(method=Average()),
            # One-hot encoding/One feature per value
            Continuize(multinomial_treatment=Continuize.Indicators),
            # Standardization (z-score normalization)
            Normalize(norm_type=Normalize.NormalizeBySD)
        )
    )

    # TODO: somehow verify that the preprocessor is working correctly (e.g. check for NaNs, check that features are continuous, etc.)
    # data = preprocessor(data)

    # 6) Evaluate
