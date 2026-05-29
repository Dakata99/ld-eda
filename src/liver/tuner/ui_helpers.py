from __future__ import annotations

import json
import math
from typing import Any

from PySide6.QtWidgets import QFrame


def vline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.VLine)
    line.setObjectName("vline")
    line.setFixedWidth(1)
    return line


def hline(section: bool = False) -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setObjectName("hline-section" if section else "hline")
    line.setFixedHeight(1)
    return line


def json_text(value: Any) -> str:
    """Display/save typed values as JSON: false, null, [100], \"l2\", 1.0."""
    return json.dumps(value)


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "estimating..."

    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def parse_manual_list(raw: str, *, learner_name: str, param_name: str) -> list[Any]:
    """Manual values are always a JSON list of candidate values."""
    raw = raw.strip()
    if not raw:
        raise ValueError(f"Manual values for {learner_name}.{param_name} are empty.")

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Manual values for {learner_name}.{param_name} must be valid JSON. "
            "Example: [0.1, 1.0, 10.0]"
        ) from exc

    if not isinstance(value, list):
        raise ValueError(
            f"Manual values for {learner_name}.{param_name} must be a JSON list. "
            f"You entered: {raw}"
        )

    if len(value) == 0:
        raise ValueError(f"Manual values for {learner_name}.{param_name} cannot be an empty list.")

    return value


def parameter_display_label(api_param_name: str, spec: dict[str, Any]) -> str:
    """Prefer Orange GUI label; fall back to the Python API parameter name."""
    return spec.get("orange_opt") or api_param_name
