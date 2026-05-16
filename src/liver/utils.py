from pathlib import Path

def root(*args):
    """TODO: write docstring.
    TODO: maybe find a better way? --- IGNORE ---

    Args:
        *args: path components to join with the project root.

    Returns:
        Path: the path to the project root joined with the provided path components.
    """
    # TODO: maybe find a better way?
    return Path(__file__).resolve().parents[2].joinpath(*args)
