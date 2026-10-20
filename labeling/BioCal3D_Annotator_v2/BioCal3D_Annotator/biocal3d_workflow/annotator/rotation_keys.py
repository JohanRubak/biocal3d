"""Space-to-rotate from any viewport control, including numeric editors."""
from PySide6 import QtCore,QtWidgets,QtGui

class SpaceRotateFilter(QtCore.QObject):
    def __init__(self,view):
        super().__init__(view);self.view=view
        QtWidgets.QApplication.instance().installEventFilter(self)

    def eligible(self,obj):
        view=self.view;window=view.window()
        if not isinstance(obj,QtWidgets.QWidget) or obj.window() is not window:return False
        focus=QtWidgets.QApplication.focusWidget()
        # Notes and free-text filters retain typing spaces; hovering the viewport
        # switches Space to rotation without requiring a mouse click.
        spin=isinstance(focus,QtWidgets.QAbstractSpinBox) or (focus is not None and isinstance(focus.parentWidget(),QtWidgets.QAbstractSpinBox))
        text=isinstance(focus,(QtWidgets.QLineEdit,QtWidgets.QPlainTextEdit,QtWidgets.QTextEdit)) and not spin
        if text:
            local=view.mapFromGlobal(QtGui.QCursor.pos())
            if not view.rect().contains(local):return False
        return True

    def eventFilter(self,obj,event):
        view=self.view;kind=event.type()
        if kind==QtCore.QEvent.Type.ApplicationDeactivate or (kind==QtCore.QEvent.Type.WindowDeactivate and obj is view.window()):
            view.set_space_down(False);return False
        if kind not in (QtCore.QEvent.Type.ShortcutOverride,QtCore.QEvent.Type.KeyPress,QtCore.QEvent.Type.KeyRelease):return False
        if event.key()!=QtCore.Qt.Key.Key_Space:return False
        if not isinstance(obj,QtWidgets.QWidget) or obj.window() is not view.window():return False
        if kind==QtCore.QEvent.Type.KeyRelease and view.space_down:
            if not event.isAutoRepeat():view.set_space_down(False)
            return True
        modified=event.modifiers()&(QtCore.Qt.KeyboardModifier.ControlModifier|QtCore.Qt.KeyboardModifier.AltModifier|QtCore.Qt.KeyboardModifier.MetaModifier)
        if modified or not self.eligible(obj):return False
        if kind==QtCore.QEvent.Type.ShortcutOverride:event.accept();return True
        if kind==QtCore.QEvent.Type.KeyPress:
            if not event.isAutoRepeat():
                view.setFocus(QtCore.Qt.FocusReason.ShortcutFocusReason);view.set_space_down(True)
            return True
        return False
