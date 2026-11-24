"""Shared Studio appearance, based on the surface annotator's dark theme."""
from PySide6 import QtGui

BACKGROUND = '#111923'
VIEW_BACKGROUND = '#131821'
FOREGROUND = '#dfe8f5'
MUTED = '#9daec4'
BORDER = '#344359'

STYLE = '''
QWidget {background:#111923;color:#dfe8f5;font-size:13px;}
QLabel {background:transparent;}
QLineEdit,QPlainTextEdit,QTextEdit,QListWidget,QComboBox,QAbstractSpinBox {
    background:#192534;border:1px solid #344359;border-radius:5px;padding:6px;}
QPushButton,QToolButton {background:#24354a;border:1px solid #3c516b;border-radius:5px;padding:9px;}
QPushButton:hover,QToolButton:hover {background:#314a65;}
QPushButton:pressed,QToolButton:pressed,QToolButton:checked {background:#304c6b;}
QPushButton#primary {background:#16675b;border-color:#269d89;}
QPushButton#primary:hover {background:#238574;border-color:#54c7ae;}
QPushButton[roiControl="true"] {padding:4px 10px;}
QLabel#roiStatus,QLabel#labelStatus {color:#c4cfde;padding:4px 8px;border:1px solid #344359;border-radius:5px;}
QListWidget::item {padding:8px;}
QListWidget::item:selected {background:#304c6b;color:#edf4ff;}
QListWidget::item:hover:!selected {background:#24354a;}
QSlider::groove:horizontal {height:5px;background:#344359;}
QSlider::handle:horizontal {width:12px;background:#6ed6bc;margin:-4px 0;border-radius:6px;}
QGroupBox {border:1px solid #344359;border-radius:6px;margin-top:12px;padding-top:10px;}
QGroupBox::title {subcontrol-origin:margin;left:12px;padding:0 5px;}
QToolTip {background:#24354a;color:#edf4ff;border:1px solid #3c516b;padding:6px;}
QMenu,QComboBox QAbstractItemView {background:#192534;color:#dfe8f5;selection-background-color:#304c6b;}
QCheckBox::indicator {width:13px;height:13px;}
QWidget:disabled {color:#788a9f;}
QPushButton:disabled,QToolButton:disabled {background:#192534;border-color:#2b384b;}
QStatusBar {color:#9daec4;border-top:1px solid #344359;}
QScrollArea {border:0;}
QFrame#studioSidebar {background:#111923;border-right:1px solid #344359;}
QListWidget#studioNav {background:transparent;border:0;font-size:14px;}
QListWidget#studioNav::item {padding:16px 10px;border-radius:6px;}
QListWidget#studioNav::item:selected {background:#304c6b;color:#edf4ff;}
'''


def palette():
    p=QtGui.QPalette()
    for role,color in [('Window',BACKGROUND),('WindowText',FOREGROUND),('Base','#192534'),
                       ('AlternateBase','#24354a'),('Text',FOREGROUND),('Button','#24354a'),
                       ('ButtonText',FOREGROUND),('Highlight','#304c6b'),('HighlightedText','#edf4ff'),
                       ('ToolTipBase','#24354a'),('ToolTipText',FOREGROUND),('PlaceholderText',MUTED),
                       ('Light','#3c516b'),('Mid',BORDER),('Dark',BACKGROUND)]:
        p.setColor(getattr(QtGui.QPalette.ColorRole,role),QtGui.QColor(color))
    for role in ('WindowText','Text','ButtonText'):
        p.setColor(QtGui.QPalette.ColorGroup.Disabled,getattr(QtGui.QPalette.ColorRole,role),QtGui.QColor('#788a9f'))
    return p


def apply_theme(widget):
    widget.setPalette(palette());widget.setStyleSheet(STYLE)
