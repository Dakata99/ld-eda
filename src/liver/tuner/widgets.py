from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QRadioButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .ui_helpers import hline, json_text, parameter_display_label, parse_manual_list, vline


@dataclass
class ParamBinding:
    learner_key: str
    learner_display_name: str
    api_param_name: str
    spec: dict[str, Any]
    manual_input: QLineEdit
    btn_group: QButtonGroup
    default_label: QLabel

    @property
    def default_value(self) -> Any:
        return self.spec["default"]

    def mode(self) -> str:
        return "manual" if self.btn_group.checkedId() == 1 else "default"

    def value(self) -> Any:
        if self.mode() == "default":
            return self.default_value
        return parse_manual_list(
            self.manual_input.text(),
            learner_name=self.learner_key,
            param_name=self.api_param_name,
        )

    def selected_values_for_sweep(self) -> list[Any]:
        if self.mode() == "manual":
            return self.value()
        return [self.default_value]

    def as_config(self) -> dict[str, Any]:
        """Serializable state. The key in the parent dict remains the API name."""
        return {
            "mode": self.mode(),
            "value": self.value(),
            "api_param": self.api_param_name,
            "orange_opt": self.spec.get("orange_opt"),
            "exposed_in_orange": self.spec.get("exposed_in_orange", False),
            "type": self.spec.get("type", "unknown"),
        }

    def apply_mode(self, mode: str) -> None:
        is_manual = mode == "manual"
        self.btn_group.button(1 if is_manual else 0).setChecked(True)
        self.default_label.setEnabled(not is_manual)
        self.manual_input.setEnabled(is_manual)

    def apply_value(self, value: Any) -> None:
        # Manual value must be a list. If a scalar is loaded from an old config,
        # wrap it so the UI still works.
        if isinstance(value, list):
            self.manual_input.setText(json_text(value))
        else:
            self.manual_input.setText(json_text([value]))


def make_section_header() -> QWidget:
    left_label = QLabel("DEFAULT")
    left_label.setObjectName("section-label")
    left_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    right_label = QLabel("MANUAL VALUES")
    right_label.setObjectName("section-label")
    right_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    left = QHBoxLayout()
    left.setContentsMargins(8, 0, 8, 0)
    left.addWidget(left_label)

    right = QHBoxLayout()
    right.setContentsMargins(8, 0, 8, 0)
    right.addWidget(right_label)

    left_widget = QWidget()
    left_widget.setLayout(left)
    right_widget = QWidget()
    right_widget.setLayout(right)

    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(0)
    row.addWidget(left_widget, 1)
    row.addWidget(vline())
    row.addWidget(right_widget, 1)

    widget = QWidget()
    widget.setObjectName("section-header")
    widget.setLayout(row)
    return widget


def make_param_label_block(api_param_name: str, spec: dict[str, Any]) -> QWidget:
    display_label = parameter_display_label(api_param_name, spec)
    exposed = bool(spec.get("exposed_in_orange", False))

    param_lbl = QLabel(f"{display_label}:")
    param_lbl.setObjectName("param-name")
    param_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    param_lbl.setToolTip(spec.get("description", ""))

    api_lbl = QLabel(f"api: {api_param_name}")
    api_lbl.setObjectName("api-name")

    badge = QLabel("ORANGE" if exposed else "PYTHON")
    badge.setObjectName("badge-orange" if exposed else "badge-python")
    badge.setToolTip("Visible in Orange GUI" if exposed else "Python API only / not directly exposed in Orange GUI")

    second_line = QHBoxLayout()
    second_line.setContentsMargins(0, 0, 0, 0)
    second_line.setSpacing(6)
    second_line.addWidget(api_lbl)
    second_line.addWidget(badge)
    second_line.addStretch()

    layout = QVBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(1)
    layout.addWidget(param_lbl)
    layout.addLayout(second_line)

    block = QWidget()
    block.setObjectName("param-label-block")
    block.setLayout(layout)
    block.setFixedWidth(240)
    return block


def make_param_row(
    learner_key: str,
    learner_display_name: str,
    api_param_name: str,
    spec: dict[str, Any],
    param_bindings: dict[str, dict[str, ParamBinding]],
) -> QWidget:
    default_value = spec["default"]
    manual_values = spec["manual_values"]

    btn_group = QButtonGroup()
    rb_default = QRadioButton("")
    rb_default.setToolTip("Use the typed default value")
    rb_manual = QRadioButton("")
    rb_manual.setToolTip("Use the JSON list from the manual side")
    btn_group.addButton(rb_default, 0)
    btn_group.addButton(rb_manual, 1)

    default_label = QLabel(json_text(default_value))
    default_label.setObjectName("default-value")
    default_label.setToolTip(f"Python value type: {type(default_value).__name__}")

    manual_input = QLineEdit()
    manual_input.setText(json_text(manual_values))
    manual_input.setPlaceholderText("JSON list, e.g. [0.1, 1.0, 10.0]")
    manual_input.setToolTip(
        "Manual mode always expects a JSON list. "
        "For list-valued parameters, use a list of lists, e.g. [[100], [50, 50]]."
    )
    manual_input.setEnabled(False)

    label_block = make_param_label_block(api_param_name, spec)

    def on_mode_changed(button_id: int) -> None:
        is_default = button_id == 0
        default_label.setEnabled(is_default)
        manual_input.setEnabled(not is_default)

    btn_group.idClicked.connect(on_mode_changed)
    rb_default.setChecked(True)

    left = QHBoxLayout()
    left.setContentsMargins(8, 5, 8, 5)
    left.setSpacing(10)
    left.addWidget(rb_default)
    left.addWidget(label_block)
    left.addWidget(default_label)
    left.addStretch()

    left_widget = QWidget()
    left_widget.setLayout(left)
    left_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    right = QHBoxLayout()
    right.setContentsMargins(8, 5, 8, 5)
    right.setSpacing(10)
    right.addWidget(rb_manual)
    right.addWidget(manual_input)
    right.addStretch()

    right_widget = QWidget()
    right_widget.setLayout(right)
    right_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(0)
    row.addWidget(left_widget, 1)
    row.addWidget(vline())
    row.addWidget(right_widget, 1)

    row_widget = QWidget()
    row_widget.setObjectName("param-row")
    row_widget.setLayout(row)

    # Keep the button group alive. No worker thread owns this object.
    btn_group.setParent(row_widget)

    param_bindings.setdefault(learner_key, {})[api_param_name] = ParamBinding(
        learner_key=learner_key,
        learner_display_name=learner_display_name,
        api_param_name=api_param_name,
        spec=spec,
        manual_input=manual_input,
        btn_group=btn_group,
        default_label=default_label,
    )

    return row_widget


def make_learner_block(
    learner_key: str,
    learner_spec: dict[str, Any],
    param_bindings: dict[str, dict[str, ParamBinding]],
) -> QWidget:
    display_name = learner_spec.get("display_name", learner_key)
    api_class = learner_spec.get("api_class", "")

    outer = QVBoxLayout()
    outer.setContentsMargins(14, 12, 14, 14)
    outer.setSpacing(0)

    title = QLabel(f"{learner_key} — {display_name}")
    title.setObjectName("learner-title")
    title.setToolTip(f"Python API class: {api_class}")
    outer.addWidget(title)
    outer.addWidget(hline())
    outer.addSpacing(4)

    outer.addWidget(make_section_header())
    outer.addWidget(hline())

    params = learner_spec.get("params", {})
    for i, (api_param_name, param_spec) in enumerate(params.items()):
        outer.addWidget(
            make_param_row(
                learner_key=learner_key,
                learner_display_name=display_name,
                api_param_name=api_param_name,
                spec=param_spec,
                param_bindings=param_bindings,
            )
        )
        if i < len(params) - 1:
            outer.addWidget(hline())

    block_widget = QWidget()
    block_widget.setObjectName("learner-block")
    block_widget.setLayout(outer)
    block_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return block_widget
