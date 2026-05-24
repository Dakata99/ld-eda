from .experiment import main as experiment
from .plot import main as plot
from .cli import CSV_FILE, HTML_FILE


def run_analysis(exprid: int, learners_group: list, config: str = "default") -> None:
	# Run the experiment
	experiment(exprid, learners_group, config)

	# Plot the results
	plot(
		exprid,
		CSV_FILE.format(experiment=exprid, config=config),
		HTML_FILE.format(experiment=exprid, config=config),
	)
