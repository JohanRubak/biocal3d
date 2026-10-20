"""Modal loading feedback with worker IO, real byte progress and elapsed time."""
import threading,time
from PySide6 import QtCore,QtWidgets

class _WorkerSignals(QtCore.QObject):
    completed=QtCore.Signal(object,object)
    progress=QtCore.Signal(str,object,object)

class LoadingFeedback(QtWidgets.QDialog):
    def __init__(self,owner,title):
        super().__init__(owner)
        self.owner=owner;self.started=time.perf_counter();self.current_stage='';self._loop=None
        self.setWindowTitle(title);self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowCloseButtonHint,False)
        self.setMinimumWidth(440)
        layout=QtWidgets.QVBoxLayout(self);layout.setContentsMargins(22,20,22,20);layout.setSpacing(12)
        heading=QtWidgets.QLabel(title);heading.setStyleSheet('font-size:17px;font-weight:600;');layout.addWidget(heading)
        self.message=QtWidgets.QLabel('Starting…');self.message.setWordWrap(True);layout.addWidget(self.message)
        self.bar=QtWidgets.QProgressBar();self.bar.setRange(0,0);self.bar.setTextVisible(False);layout.addWidget(self.bar)
        self.elapsed=QtWidgets.QLabel('Elapsed 0.0 s');layout.addWidget(self.elapsed)
        self.timer=QtCore.QTimer(self);self.timer.setInterval(100);self.timer.timeout.connect(self.update_elapsed)
        self.signals=_WorkerSignals(self);self.signals.completed.connect(self._complete,QtCore.Qt.ConnectionType.QueuedConnection)
        self.signals.progress.connect(self.stage,QtCore.Qt.ConnectionType.QueuedConnection)
        self._last_progress=0.;self.stage_timings=[]
    def __enter__(self):
        self.owner._loading_depth=getattr(self.owner,'_loading_depth',0)+1
        app=QtWidgets.QApplication.instance()
        self._previous_window=app.activeWindow();self._quit_on_last_window=app.quitOnLastWindowClosed()
        app.setQuitOnLastWindowClosed(False)
        self.show();self.timer.start();self.stage('Starting…');return self
    def __exit__(self,*args):
        self.timer.stop();self.hide();self.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        self.owner._loading_depth=max(0,getattr(self.owner,'_loading_depth',1)-1)
        app=QtWidgets.QApplication.instance()
        try:
            previous=self._previous_window
            if previous is not None and previous.isVisible():
                previous.activateWindow();app.setActiveWindow(previous)
        except RuntimeError:pass
        app.setQuitOnLastWindowClosed(self._quit_on_last_window)
        self.deleteLater()
        # Navigation is synchronous to callers. Retire the modal window now so
        # an immediate next A/D key is not blocked by a deferred dialog deletion.
        QtCore.QCoreApplication.sendPostedEvents(self,QtCore.QEvent.Type.DeferredDelete)
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
    def reject(self):pass
    def closeEvent(self,event):event.ignore()
    def update_elapsed(self):self.elapsed.setText(f'Elapsed {time.perf_counter()-self.started:.1f} s')
    @QtCore.Slot(str,object,object)
    def stage(self,message,done=None,total=None):
        self.current_stage=message;self.message.setText(message);self.update_elapsed()
        if total is not None and total>0 and done is not None:
            self.bar.setRange(0,1000);self.bar.setValue(min(1000,int(1000*done/total)))
            self.bar.setFormat(f'{done/1024**2:.1f} / {total/1024**2:.1f} MB' if total>100000 else f'{done} / {total}')
            self.bar.setTextVisible(True)
        else:self.bar.setRange(0,0);self.bar.setTextVisible(False)
        # Paint feedback before a short main-thread VTK preparation step.
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
    def report(self,message,done=None,total=None):
        now=time.perf_counter()
        if now-self._last_progress>.08 or (total is not None and done==total):
            self._last_progress=now;self.signals.progress.emit(message,done,total)
    @QtCore.Slot(object,object)
    def _complete(self,value,error):
        self._value=value;self._error=error
        if self._loop is not None:self._loop.quit()
    def run(self,function,message):
        self.stage(message);self._value=None;self._error=None
        self._loop=QtCore.QEventLoop(self)
        def target():
            try:self.signals.completed.emit(function(),None)
            except BaseException as error:self.signals.completed.emit(None,error)
        worker=threading.Thread(target=target,name='BioCal3D-input-loader',daemon=False)
        start=time.perf_counter();worker.start();self._loop.exec();worker.join()
        self._loop.deleteLater();self._loop=None
        self.stage_timings.append((message,time.perf_counter()-start))
        if self._error is not None:raise self._error
        return self._value


class SilentLoading:
    """Reuse input preparation and timings without a popup or nested event loop.

    Sample navigation is synchronous, so queued clicks/painting cannot modify
    the previous surface while the replacement is being prepared.
    """
    def __init__(self,owner):
        self.owner=owner;self.stage_timings=[]
    def __enter__(self):
        self.owner._loading_depth=getattr(self.owner,'_loading_depth',0)+1
        return self
    def __exit__(self,*args):
        self.owner._loading_depth=max(0,getattr(self.owner,'_loading_depth',1)-1)
    def stage(self,message,done=None,total=None):pass
    def report(self,message,done=None,total=None):pass
    def run(self,function,message):
        started=time.perf_counter()
        try:return function()
        finally:self.stage_timings.append((message,time.perf_counter()-started))
