"""
TODO: add docstring.
Binary classification for indian dataset.
"""
from loguru import logger

from .load import load_csv

def main():
    """TODO: write docstring."""
    logger.info('=== Running experiment 2: binary classification for indian dataset ===')
    data = load_csv('indian')

    

