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
	parser.add_argument(
		"--learners-group",
		choices=[
			"logistic-regression",
			"random-forest",
			"tree",
			"gradient-boosting",
			"neural-network",
			"svm",
		],
		nargs="+",
		default=None,
		help="Run specific family(ies) of learners.",
	)
	parser.add_argument(
		"--config",
		type=str,
		choices=["default", "global", "experiment1", "experiment2", "experiment3"],
		default="default",
		help="Configuration to use for the experiment",
	)
	parser.add_argument(
		"--plot-only",
		action="store_true",
		default=False,
		help="Plot only on already existing results.",
	)
	argcomplete.autocomplete(parser)
	args = parser.parse_args()

	# Set up logging
	setup_logging(args.debug)

	if not args.plot_only:
		# Run the analysis for the specified experiment
		from .core import run_analysis

		run_analysis(args.experiment, args.learners_group, args.config)
	else:
		# Plot the results for the specified experiment
		from .plot import main as plot

		plot(
			args.experiment,
			f"experiment{args.experiment}-{args.config}-evaluation-results.csv",
			f"experiment{args.experiment}-{args.config}.html",
		)
