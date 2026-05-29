STYLESHEET = """
QWidget {
    background-color: #fdf6ee;
    color: #2c1a00;
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    font-size: 13px;
}

QLabel#learner-title {
    font-size: 15px;
    font-weight: bold;
    color: #c45000;
    letter-spacing: 1px;
    padding: 4px 0px 6px 0px;
}

QLabel#section-label {
    color: #7a4a1e;
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 1px;
    padding: 4px 0px;
}

QLabel#param-name {
    color: #7a4a1e;
    font-size: 13px;
    padding-left: 6px;
}

QLabel#api-name {
    color: #a06030;
    font-size: 11px;
    font-family: 'Consolas', 'Fira Code', monospace;
    padding-left: 6px;
}

QLabel#badge-orange {
    color: #7c2d00;
    background-color: #ffe5c8;
    border: 1px solid #fdba74;
    border-radius: 4px;
    padding: 1px 5px;
    font-size: 10px;
    font-weight: bold;
}

QLabel#badge-python {
    color: #6b3b16;
    background-color: #f5ece0;
    border: 1px solid #ddc9ae;
    border-radius: 4px;
    padding: 1px 5px;
    font-size: 10px;
    font-weight: bold;
}

QLabel#default-value {
    color: #7c2d00;
    font-weight: bold;
    font-size: 13px;
    padding: 2px 8px;
    background-color: #ffe5c8;
    border: 1px solid #f97316;
    border-radius: 4px;
    font-family: 'Consolas', 'Fira Code', monospace;
}

QLabel#default-value:disabled {
    color: #c4a882;
    background-color: #f5ece0;
    border-color: #ddc9ae;
}

QLineEdit {
    background-color: #fff8f0;
    color: #2c1a00;
    border: 1px solid #fdba74;
    border-radius: 5px;
    padding: 4px 10px;
    font-family: 'Consolas', 'Fira Code', monospace;
    font-size: 12px;
    min-width: 220px;
}

QLineEdit:focus {
    border-color: #ea580c;
    background-color: #fff3e6;
}

QLineEdit:disabled {
    color: #c4a882;
    background-color: #f5ece0;
    border-color: #e8d5bb;
}

QRadioButton {
    color: #a35c2a;
    spacing: 6px;
    font-size: 12px;
}

QRadioButton::indicator {
    width: 14px;
    height: 14px;
    border-radius: 7px;
    border: 2px solid #fdba74;
    background-color: #fff8f0;
}

QRadioButton::indicator:checked {
    background-color: #ea580c;
    border-color: #ea580c;
}

QRadioButton::indicator:hover {
    border-color: #ea580c;
}

QFrame#vline {
    color: #fcd9b0;
    background-color: #fcd9b0;
    max-width: 1px;
    min-width: 1px;
}

QFrame#hline {
    color: #fcd9b0;
    background-color: #fcd9b0;
    max-height: 1px;
    min-height: 1px;
}

QFrame#hline-section {
    color: #f4c08a;
    background-color: #f4c08a;
    max-height: 1px;
    min-height: 1px;
}

QWidget#learner-block {
    background-color: #fff8f0;
    border: 1px solid #fdba74;
    border-radius: 10px;
}

QWidget#param-row, QWidget#section-header, QWidget#status-panel, QWidget#param-label-block {
    background-color: transparent;
}

QWidget#toolbar {
    background-color: #fff0dc;
    border: 1px solid #fdba74;
    border-radius: 10px;
}

QLabel#toolbar-label {
    color: #7a4a1e;
    font-size: 12px;
    font-weight: bold;
    letter-spacing: 0.5px;
}

QLabel#status-label {
    color: #7a4a1e;
    font-size: 12px;
    font-weight: bold;
}

QLabel#time-label {
    color: #a06030;
    font-size: 12px;
    font-family: 'Consolas', 'Fira Code', monospace;
}

QProgressBar {
    background-color: #fff8f0;
    border: 1px solid #fdba74;
    border-radius: 6px;
    text-align: center;
    color: #7c2d00;
    font-weight: bold;
    min-height: 18px;
}

QProgressBar::chunk {
    background-color: #ea580c;
    border-radius: 5px;
}

QPushButton {
    font-size: 13px;
    font-weight: bold;
    border-radius: 6px;
    padding: 6px 18px;
    border: none;
}

QPushButton#btn-secondary {
    background-color: #ffe5c8;
    color: #7c2d00;
    border: 1px solid #fdba74;
}

QPushButton#btn-secondary:hover {
    background-color: #ffd5aa;
    border-color: #ea580c;
}

QPushButton#btn-secondary:pressed {
    background-color: #ffc890;
}

QPushButton#btn-run {
    background-color: #ea580c;
    color: #ffffff;
    padding: 7px 32px;
    font-size: 14px;
    letter-spacing: 1px;
}

QPushButton#btn-run:hover {
    background-color: #c2410c;
}

QPushButton#btn-run:pressed {
    background-color: #9a3412;
}

QPushButton#btn-run:disabled, QPushButton#btn-secondary:disabled {
    background-color: #f1dcc4;
    color: #b99775;
    border-color: #e8d5bb;
}

QRadioButton#method-radio {
    color: #7a4a1e;
    font-size: 13px;
    spacing: 6px;
}

QRadioButton#method-radio:checked {
    color: #7c2d00;
    font-weight: bold;
}

QRadioButton#method-radio::indicator {
    width: 14px;
    height: 14px;
    border-radius: 7px;
    border: 2px solid #fdba74;
    background-color: #fff8f0;
}

QRadioButton#method-radio::indicator:checked {
    background-color: #ea580c;
    border-color: #ea580c;
}

QRadioButton#method-radio::indicator:hover {
    border-color: #ea580c;
}

QScrollArea {
    border: none;
    background-color: #fdf6ee;
}
"""
