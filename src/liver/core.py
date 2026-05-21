# from .experiment1 import main as experiment1
# from .experiment2 import main as experiment2
# from .experiment3 import main as experiment3
from .experiment import main as experiment
from .plot2 import main as plot


def run_analysis(exprid: int, learner_group: str = "all", config: str = "global") -> None:
	# # Run the experiment
	# if exprid == 1:
	#     experiment1(learner_group, config)
	# elif exprid == 2:
	#     experiment2(learner_group, config)
	# else:
	#     experiment3(learner_group, config)
	# plot(exprid, f"experiment{exprid}-evaluation-results.csv", f'experiment{exprid}.html')

	# Run the experiment
	experiment(exprid, learner_group, config)
	# Plot the results
	plot(exprid, f"new-experiment{exprid}-evaluation-results.csv", f"new-experiment{exprid}.html")
