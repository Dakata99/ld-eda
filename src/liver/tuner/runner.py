from __future__ import annotations

import itertools
import math
import threading
import time
from typing import Any

from AnyQt.QtCore import QThread, Signal, QEvent


def configured_values_for_sweep(info: dict[str, Any]) -> list[Any]:
    """
    Default mode gives one value. Manual mode gives many values.

    Important example:
    - default hidden_layer_sizes = [100] means one configuration: [100]
    - manual hidden_layer_sizes = [[100], [50, 50]] means two configurations
    """
    if info["mode"] == "manual":
        return info["value"]
    return [info["value"]]


def estimate_total_work_units(cfg: dict[str, Any]) -> int:
    method_multiplier = 5 if cfg.get("method") == "cross_validation" else 1
    total = 0

    for learner_info in cfg.get("learners", {}).values():
        params = learner_info.get("params", learner_info)
        lengths = [len(configured_values_for_sweep(info)) for info in params.values()]
        learner_configs = math.prod(lengths) if lengths else 1
        total += learner_configs * method_multiplier

    return max(1, total)


class RunThread(QThread):
    """
    Worker thread with no QWidget/QObject children.

    Rule of thumb:
    - worker thread may compute and emit signals;
    - main thread updates widgets;
    - do not create, edit, parent, or delete UI widgets here.
    """

    progress_changed = Signal(int, int, float, object, str)
    run_finished = Signal(bool, str)

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        total = estimate_total_work_units(self.cfg)
        current = 0
        start = time.monotonic()

        from Orange.data import Table

        from liver.experiment import TestAndScore
        from liver.load import load_configuration
        from liver.utils import create_learners, root

        try:
            config = load_configuration('default')
            learners = create_learners(config)

            train = Table(str(root("datasets", "expr1", "expr1-train-data.tab")))
            test = Table(str(root("datasets", "expr1", "expr1-test-data.tab")))
            ts = TestAndScore(learners)
            ts.train(train, test, 'totd')
            # ts.eval(exprid, CSV_FILE.format(experiment=exprid, config=configuration, method=method))

            # method = self.cfg.get("method", "cross_validation")
            # folds = 5 if method == "cross_validation" else 1

            # for learner_key, learner_info in self.cfg.get("learners", {}).items():
            #     params = learner_info.get("params", {})
            #     display_name = learner_info.get("display_name", learner_key)
            #     api_class = learner_info.get("api_class", learner_key)

            #     param_names = list(params.keys())
            #     sweep_values = [configured_values_for_sweep(params[p]) for p in param_names]

            #     for combo in itertools.product(*sweep_values):
            #         learner_kwargs = dict(zip(param_names, combo, strict=True))

            #         for fold_idx in range(1, folds + 1):
            #             if self._cancel_event.is_set():
            #                 self.run_finished.emit(False, "Run cancelled safely.")
            #                 return

            #             message = (
            #                 f"{display_name} ({api_class}) | "
            #                 f"kwargs={learner_kwargs} | step {fold_idx}/{folds}"
            #             )

            #             # --------------------------------------------------------------
            #             # TODO: IMPLEMENT YOUR REAL "RUN" FUNCTIONALITY HERE.
            #             #
            #             # At this point you already have correct Python API kwargs.
            #             # Example:
            #             #
            #             #   learner = LogisticRegressionLearner(**learner_kwargs)
            #             #
            #             # For the Balance class distribution example, the GUI label is
            #             # shown as "Balance class distribution", but the kwarg is:
            #             #
            #             #   {"class_weight": "balanced"}
            #             #
            #             # Keep the work chunked. Check self._cancel_event.is_set()
            #             # between folds/configurations so Ctrl+C and Cancel can stop
            #             # cleanly after the current chunk completes.
            #             # --------------------------------------------------------------
            #             time.sleep(0.20)

            current += 1
            elapsed = time.monotonic() - start
            remaining = None
            if current > 0:
                average_per_unit = elapsed / current
                remaining = max(0.0, average_per_unit * (total - current))

            self.progress_changed.emit(current, total, elapsed, remaining, 'AAAAA')

            self.run_finished.emit(True, "Run finished successfully.")

        except Exception as exc:  # pragma: no cover - shown in GUI during normal use
            self.run_finished.emit(False, f"Run failed: {exc}")
