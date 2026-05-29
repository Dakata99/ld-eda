import argparse

import argcomplete

AVAILABLE_CONFIGS = (
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
	parser.add_argument("--experiment", type=int, choices=[1, 2, 3])
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
	pt = parser.add_mutually_exclusive_group()
	pt.add_argument(
		"--plot-only",
		action="store_true",
		default=False,
		help="Plot only on already existing results.",
	)
	pt.add_argument(
		"--iplot",
		action="store_true",
		default=False,
		help="Plot only on already existing results (interactivity).",
	)
	mt = parser.add_mutually_exclusive_group()
	mt.add_argument(
		'--cross-validation',
		action="store_true",
		default=False,
		help='Use Cross Validation method.'
	)
	mt.add_argument(
		'--hold-out',
		action="store_true",
		default=True,
		help='Use Hold-out (TestOnTestData) method.'
	)
	argcomplete.autocomplete(parser)
	args = parser.parse_args()

	# Set up logging
	setup_logging(args.debug)

	# Method for evaluation, hold-out = test on test data
	method = 'cv' if args.cross_validation else 'totd'

	if args.experiment and not args.plot_only and not args.iplot:
		# Run the analysis for the specified experiment
		from .core import run_analysis

		run_analysis(args.experiment, method, args.learners_group, args.config)
	elif args.plot_only:
		# Plot the results for the specified experiment
		from .plot import main as plot

		plot(args.experiment, method, args.config)
	elif args.iplot:
		from pathlib import Path

		from .plot import main as plot
		from .utils import root

		# Get available CSV files
		csvs: list[Path] = list(root('results').relative_to(root()).glob('*.csv'))
		csvs.sort()

		prompt = "CSV files:\n"
		for i, csv in enumerate(csvs, 1):
			prompt += f'{i}) {csv}\n'
		prompt += '\nChoose CSV file to plot (enter a number): '

		idx = int(input(prompt))
		csv: Path = csvs[idx - 1]
		filename: str = csv.stem
		parts = filename.split('-')
		exprid, config, method = int(parts[0][-1]), parts[1], parts[2]

		# Plot the results for the specified experiment
		plot(exprid, method, config)
