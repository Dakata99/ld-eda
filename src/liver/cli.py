import argparse
import argcomplete


def setup_logging(debug: bool = False) -> None:
    import sys
    from loguru import logger

    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if debug else "INFO",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    # Set up logging
    setup_logging(args.debug)

    # Run the analysis for the specified experiment
    from .core import run_analysis

    run_analysis(args.experiment)
