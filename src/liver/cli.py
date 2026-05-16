import argparse
import argcomplete
from .core import run_analysis


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int, required=True, choices=[1, 2, 3])
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    run_analysis(args.experiment)
