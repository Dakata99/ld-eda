from __future__ import annotations

import sys

from AnyQt.QtWidgets import QApplication

from .tuner.app_window import MainWindow
from .tuner.styles import STYLESHEET


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
