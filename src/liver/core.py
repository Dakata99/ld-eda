from .experiment1 import main as experiment1
from .experiment2 import main as experiment2
from .experiment3 import main as experiment3
from .plot2 import main as plot


def run_analysis(experiment: int, learner_group: str = "all", config: str = "global") -> None:
    # Run the experiment
    if experiment == 1:
        experiment1(learner_group, config)
    elif experiment == 2:
        experiment2(learner_group, config)
    else:
        experiment3(learner_group, config)

    # Plot the results
    plot(experiment)
