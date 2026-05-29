from __future__ import annotations

import json
import signal
from typing import Any

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .config import LEARNER_PAIRS, LEARNER_SPECS, LearnerSpecs
from .runner import RunThread, estimate_total_work_units
from .ui_helpers import format_duration, hline, vline
from .widgets import ParamBinding, make_learner_block


class MainWindow(QWidget):
    def __init__(self, learner_specs: LearnerSpecs | None = None) -> None:
        super().__init__()
        self.learner_specs = learner_specs or LEARNER_SPECS
        self.param_bindings: dict[str, dict[str, ParamBinding]] = {}
        self.run_thread: RunThread | None = None
        self.is_running = False
        self.is_cancelling = False

        self.setWindowTitle("Learner Parameter Tuner")
        self.setMinimumWidth(1180)

        self._build_ui()
        self._connect_signals()
        self._install_sigint_handler()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(14)

        header = QLabel("Learner Parameter Tuner")
        header.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        header.setStyleSheet("color: #c45000; letter-spacing: 1px; padding-bottom: 2px;")
        main_layout.addWidget(header)

        subtitle = QLabel("Configure typed defaults or JSON-list manual sweep values per learner")
        subtitle.setStyleSheet("color: #a06030; font-size: 12px; padding-bottom: 6px;")
        main_layout.addWidget(subtitle)

        main_layout.addWidget(hline(section=True))

        toolbar = self._build_toolbar()
        main_layout.addWidget(toolbar)

        status_panel = self._build_status_panel()
        main_layout.addWidget(status_panel)

        main_layout.addWidget(hline(section=True))

        self.learner_container = self._build_learner_grid()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.learner_container)
        main_layout.addWidget(scroll_area, 1)

        self.widgets_disabled_during_run = [
            self.btn_load,
            self.btn_dump,
            self.rb_cv,
            self.rb_holdout,
            self.learner_container,
        ]

        self.setLayout(main_layout)
        self.resize(1240, 780)

    def _build_toolbar(self) -> QWidget:
        toolbar_widget = QWidget()
        toolbar_widget.setObjectName("toolbar")
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(16, 10, 16, 10)
        toolbar_layout.setSpacing(10)

        cfg_label = QLabel("CONFIG")
        cfg_label.setObjectName("toolbar-label")
        toolbar_layout.addWidget(cfg_label)

        self.btn_load = QPushButton("⬆  Load")
        self.btn_load.setObjectName("btn-secondary")
        self.btn_load.setToolTip("Load a saved configuration from a JSON file")

        self.btn_dump = QPushButton("⬇  Dump")
        self.btn_dump.setObjectName("btn-secondary")
        self.btn_dump.setToolTip("Save the current configuration to a JSON file")

        toolbar_layout.addWidget(self.btn_load)
        toolbar_layout.addWidget(self.btn_dump)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.VLine)
        sep1.setObjectName("vline")
        sep1.setFixedHeight(28)
        toolbar_layout.addSpacing(6)
        toolbar_layout.addWidget(sep1)
        toolbar_layout.addSpacing(6)

        method_label = QLabel("METHOD")
        method_label.setObjectName("toolbar-label")
        toolbar_layout.addWidget(method_label)

        self.method_group = QButtonGroup(self)
        self.rb_cv = QRadioButton("Cross-Validation")
        self.rb_cv.setObjectName("method-radio")
        self.rb_holdout = QRadioButton("Hold-Out")
        self.rb_holdout.setObjectName("method-radio")
        self.method_group.addButton(self.rb_cv, 0)
        self.method_group.addButton(self.rb_holdout, 1)
        self.rb_cv.setChecked(True)

        toolbar_layout.addWidget(self.rb_cv)
        toolbar_layout.addWidget(self.rb_holdout)
        toolbar_layout.addStretch()

        self.btn_run = QPushButton("▶  RUN")
        self.btn_run.setObjectName("btn-run")
        self.btn_run.setToolTip("Run with the current configuration")
        toolbar_layout.addWidget(self.btn_run)

        toolbar_widget.setLayout(toolbar_layout)
        return toolbar_widget

    def _build_status_panel(self) -> QWidget:
        status_panel = QWidget()
        status_panel.setObjectName("status-panel")
        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(12)

        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("status-label")
        self.status_label.setMinimumWidth(260)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0%")

        self.elapsed_label = QLabel("Elapsed: 00:00")
        self.elapsed_label.setObjectName("time-label")
        self.elapsed_label.setMinimumWidth(120)

        self.remaining_label = QLabel("Remaining: --:--")
        self.remaining_label.setObjectName("time-label")
        self.remaining_label.setMinimumWidth(160)

        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar, 1)
        status_layout.addWidget(self.elapsed_label)
        status_layout.addWidget(self.remaining_label)
        status_panel.setLayout(status_layout)
        return status_panel

    def _build_learner_grid(self) -> QWidget:
        learner_container = QWidget()
        learner_layout = QVBoxLayout()
        learner_layout.setContentsMargins(0, 0, 0, 0)
        learner_layout.setSpacing(14)

        for r, (left_key, right_key) in enumerate(LEARNER_PAIRS):
            row = QHBoxLayout()
            row.setSpacing(14)
            row.addWidget(make_learner_block(left_key, self.learner_specs[left_key], self.param_bindings))
            row.addWidget(make_learner_block(right_key, self.learner_specs[right_key], self.param_bindings))

            row_widget = QWidget()
            row_widget.setLayout(row)
            learner_layout.addWidget(row_widget)

            if r < len(LEARNER_PAIRS) - 1:
                learner_layout.addWidget(hline(section=True))

        learner_layout.addStretch()
        learner_container.setLayout(learner_layout)
        return learner_container

    # ------------------------------------------------------------------
    # Signals and runtime behavior
    # ------------------------------------------------------------------
    def _connect_signals(self) -> None:
        self.btn_load.clicked.connect(self.on_load)
        self.btn_dump.clicked.connect(self.on_dump)
        self.btn_run.clicked.connect(self.on_run_clicked)

    def _install_sigint_handler(self) -> None:
        signal.signal(signal.SIGINT, self.handle_sigint)

        # Qt needs a tiny timer so Python gets a chance to process SIGINT/Ctrl+C.
        self.signal_timer = QTimer(self)
        self.signal_timer.start(200)
        self.signal_timer.timeout.connect(lambda: None)

    def collect_config(self) -> dict[str, Any]:
        """Return current UI state. Parent keys are learner IDs and API parameter names."""
        cfg: dict[str, Any] = {
            "schema_version": 2,
            "method": "cross_validation" if self.rb_cv.isChecked() else "hold_out",
            "learners": {},
        }

        for learner_key, params in self.param_bindings.items():
            learner_spec = self.learner_specs[learner_key]
            cfg["learners"][learner_key] = {
                "display_name": learner_spec.get("display_name", learner_key),
                "api_class": learner_spec.get("api_class"),
                "orange_widget": learner_spec.get("orange_widget"),
                "params": {},
            }

            for api_param_name, binding in params.items():
                cfg["learners"][learner_key]["params"][api_param_name] = binding.as_config()

        return cfg

    def apply_config(self, cfg: dict[str, Any]) -> None:
        """Apply a loaded config dict back to the UI widgets."""
        method = cfg.get("method", "cross_validation")
        (self.rb_cv if method == "cross_validation" else self.rb_holdout).setChecked(True)

        for learner_key, learner_info in cfg.get("learners", {}).items():
            if learner_key not in self.param_bindings:
                continue

            # Supports both the new nested shape and the older direct-param shape.
            params = learner_info.get("params", learner_info)

            for api_param_name, info in params.items():
                if api_param_name not in self.param_bindings[learner_key]:
                    continue

                binding = self.param_bindings[learner_key][api_param_name]
                mode = info.get("mode", "default")
                value = info.get("value", [binding.default_value])

                if mode == "manual":
                    binding.apply_value(value)
                    binding.apply_mode("manual")
                else:
                    binding.apply_mode("default")

    def set_running_state(self, running: bool, *, cancelling: bool = False) -> None:
        self.is_running = running
        self.is_cancelling = cancelling

        for widget in self.widgets_disabled_during_run:
            widget.setEnabled(not running)

        if running:
            self.btn_run.setText("■  CANCEL" if not cancelling else "Cancelling...")
            self.btn_run.setEnabled(not cancelling)
        else:
            self.btn_run.setText("▶  RUN")
            self.btn_run.setEnabled(True)

    def on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                cfg = json.load(f)
            self.apply_config(cfg)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", f"Could not load configuration:\n{exc}")

    def on_dump(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Configuration", "config.json", "JSON Files (*.json)")
        if not path:
            return
        try:
            cfg = self.collect_config()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            QMessageBox.information(self, "Saved", f"Configuration saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", f"Could not save configuration:\n{exc}")

    def on_progress(self, current: int, total: int, elapsed: float, remaining: float | None, message: str) -> None:
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)
        percent = int((current / total) * 100) if total else 0
        self.progress_bar.setFormat(f"{percent}%  ({current}/{total})")

        self.elapsed_label.setText(f"Elapsed: {format_duration(elapsed)}")
        self.remaining_label.setText(f"Remaining: {format_duration(remaining)}")
        self.status_label.setText(message)

    def cleanup_thread(self) -> None:
        self.run_thread = None

    def on_finished(self, success: bool, message: str) -> None:
        self.set_running_state(False)
        self.status_label.setText(message)
        self.remaining_label.setText("Remaining: 00:00" if success else "Remaining: --:--")

        if success:
            QMessageBox.information(self, "Run", message)
        else:
            QMessageBox.warning(self, "Run", message)

    def start_run(self) -> None:
        try:
            cfg = self.collect_config()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid configuration", str(exc))
            return

        total = estimate_total_work_units(cfg)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(f"0%  (0/{total})")
        self.elapsed_label.setText("Elapsed: 00:00")
        self.remaining_label.setText("Remaining: estimating...")
        self.status_label.setText("Starting run...")

        self.set_running_state(True)

        self.run_thread = RunThread(cfg)
        self.run_thread.progress_changed.connect(self.on_progress)
        self.run_thread.run_finished.connect(self.on_finished)
        self.run_thread.finished.connect(self.cleanup_thread)
        self.run_thread.start()

    def cancel_run(self, reason: str = "Cancellation requested...") -> None:
        if not self.is_running or self.run_thread is None:
            return
        self.status_label.setText(reason)
        self.set_running_state(True, cancelling=True)
        self.run_thread.cancel()

    def on_run_clicked(self) -> None:
        if self.is_running:
            self.cancel_run()
        else:
            self.start_run()

    def handle_sigint(self, signum, frame) -> None:  # noqa: ANN001, ARG002
        """Handle Ctrl+C from the terminal without leaving the worker half-running."""
        if self.is_running:
            self.cancel_run("Ctrl+C received. Cancelling safely...")
        else:
            QApplication.quit()
