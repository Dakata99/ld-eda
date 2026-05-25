import argparse

import argcomplete

AVAILABLE_CONFIGS: tuple[str] = (
	"default",
	"default-full",
	"global",
	"experiment1",
	"experiment2",
	"experiment3",
)


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
		choices=AVAILABLE_CONFIGS,
		default="default",
		help="Configuration to use for the experiment",
	)
	parser.add_argument(
		"--plot-only",
		action="store_true",
		default=False,
		help="Plot only on already existing results.",
	)
	mt = parser.add_mutually_exclusive_group()
	mt.add_argument(
		'--cross-validation',
		action="store_true",
		default=False,
		help='Use CrossValidation method'
	)
	mt.add_argument(
		'--test-on-test-data',
		action="store_true",
		default=True,
		help='Use TestOnTestData method'
	)
	argcomplete.autocomplete(parser)
	args = parser.parse_args()

	# Set up logging
	setup_logging(args.debug)

	# Method for evaluation, totd = test on test data
	method = 'cv' if args.cross_validation else 'totd'

	if not args.plot_only:
		# Run the analysis for the specified experiment
		from .core import run_analysis

		run_analysis(args.experiment, method, args.learners_group, args.config)
	else:
		# Plot the results for the specified experiment
		from .plot import main as plot

		plot(args.experiment, method, args.config)
