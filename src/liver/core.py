from .experiment import main as experiment
from .plot import main as plot


def run_analysis(exprid: int, method: str, learners_group: list, config: str = "default") -> None:
	# Run the experiment
	experiment(exprid, method, learners_group, config)

	# Plot the results
	plot(exprid, method, config)
