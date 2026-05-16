"""
TODO: add docstring.
Binary classification for all 3 dataset.
"""

from loguru import logger

from .load import load_csv

def main():
    """TODO: write docstring."""
    logger.info('Running experiment 3: binary classification for all 3 datasets')

    indian = load_csv('indian')
    hcv = load_csv('hcv')
    liver = load_csv('liver')
