from .experiment import main as experiment
from .plot import main as plot


def run_analysis(exprid: int, learners_group: list, config: str = "global") -> None:
	# Run the experiment
	experiment(exprid, learners_group, config)

	# Plot the results
	plot(exprid, f"new-experiment{exprid}-evaluation-results.csv", f"new-experiment{exprid}.html")
