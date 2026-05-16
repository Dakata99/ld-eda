from .load import *

from .experiment1 import main as experiment1
from .experiment2 import main as experiment2
from .experiment3 import main as experiment3

def run_analysis(experiment: int):
    if experiment == 1:
        experiment1()
    elif experiment == 2:
        experiment2()
    else:
        experiment3()
