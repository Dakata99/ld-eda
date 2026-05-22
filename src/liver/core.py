from .experiment import main as experiment
from .plot import main as plot


def run_analysis(exprid: int, learners_group: list, config: str = "default") -> None:
	# Run the experiment
	experiment(exprid, learners_group, config)

	# Plot the results
	plot(
		exprid,
		f"experiment{exprid}-{config}-evaluation-results.csv",
		f"experiment{exprid}-{config}.html",
	)
