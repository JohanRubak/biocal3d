"""Qt orientation workspace using the original, label-preserving orientation engine."""
from pathlib import Path
import copy
import math
import uuid
import numpy as np
from PySide6 import QtCore, QtWidgets
from orientation_views import OrientationViews
import orient_scanners as core
from original_models import OriginalModelIndex
from orientation_preview import preview_model


class Signals(QtCore.QObject):
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    progress = QtCore.Signal(str)


class Task(QtCore.QRunnable):
    def __init__(self, function):
        super().__init__(); self.function = function; self.signals = Signals()
    def run(self):
        try: self.signals.done.emit(self.function())
        except Exception as exc: self.signals.failed.emit(str(exc))


class OrientationPage(QtWidgets.QWidget):
    output_ready = QtCore.Signal(str)
    dataset_changed = QtCore.Signal(str)
    originals_changed = QtCore.Signal(str)
    def __init__(self):
        super().__init__()
        from theme import apply_theme
        apply_theme(self)
        self.data_root = None; self.raw = None; self.rr = []; self.evidence = {}; self.config = None
        self.cache = {}; self.reference = None; self.source = None; self.pending = []; self.busy = False
        self.applying=False;self.before_apply=lambda:True
        self.dirty = False; self.pool = QtCore.QThreadPool(self); self.pool.setMaxThreadCount(1); self.task = None
        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel('Orient scanners to TRIOS5'); title.setStyleSheet('font-size:22px;font-weight:700'); layout.addWidget(title)
        hint = QtWidgets.QLabel('Match tooth or holder landmarks in the existing ROI frame. Rotation is applied to a new dataset copy, including its labels.'); hint.setWordWrap(True); layout.addWidget(hint)
        self.controls = QtWidgets.QWidget(); controls = QtWidgets.QVBoxLayout(self.controls); controls.setContentsMargins(0,0,0,0); layout.addWidget(self.controls)
        bar = QtWidgets.QHBoxLayout(); controls.addLayout(bar)
        for text, fn in [('Approved ROI dataset…', self.choose_dataset), ('Original models…', self.choose_raw), ('Load rotations…', self.load_saved), ('Save rotations…', self.save), ('Apply to new copy…', self.apply)]:
            b = QtWidgets.QPushButton(text); b.clicked.connect(fn); bar.addWidget(b)
            if fn==self.apply:b.setObjectName('primary')
        row = QtWidgets.QHBoxLayout(); controls.addLayout(row)
        self.ref_sample = QtWidgets.QComboBox(); self.scanner = QtWidgets.QComboBox(); self.sample = QtWidgets.QComboBox()
        for text, box in [('TRIOS5 representative',self.ref_sample), ('Scanner',self.scanner), ('Representative',self.sample)]:
            row.addWidget(QtWidgets.QLabel(text)); row.addWidget(box,1)
        self.ref_sample.currentTextChanged.connect(self.load_previews); self.scanner.currentTextChanged.connect(self.select_scanner); self.sample.currentTextChanged.connect(self.load_previews)
        self.views=OrientationViews(self);layout.addWidget(self.views,1);self.views.landmark.connect(self.pick_landmark)
        self.edit_controls = QtWidgets.QWidget(); row = QtWidgets.QHBoxLayout(self.edit_controls); layout.addWidget(self.edit_controls)
        row.addWidget(QtWidgets.QLabel('Counterclockwise rotation'))
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal); self.slider.setRange(-1800,1800); row.addWidget(self.slider,1)
        self.angle = QtWidgets.QDoubleSpinBox(); self.angle.setRange(-180,180); self.angle.setDecimals(2); self.angle.setSingleStep(.1); self.angle.setSuffix('°'); row.addWidget(self.angle)
        self.slider.valueChanged.connect(lambda n:self.angle.setValue(n/10)); self.angle.valueChanged.connect(self.angle_changed)
        for text,fn in [('Reset angle',lambda:self.angle.setValue(0)),('Top view',self.views.top),('Fit all views',self.fit),('Match landmarks',self.landmarks),('Cancel landmarks',self.cancel_landmarks),('Accept scanner',self.accept)]:
            b=QtWidgets.QPushButton(text); b.clicked.connect(fn); row.addWidget(b)
        self.note=QtWidgets.QLabel('Left drag: rotate view · Middle drag: pan · Scroll: zoom. All views are linked. The angle control rotates scanner geometry; TRIOS5 stays at 0°.'); self.note.setWordWrap(True); layout.addWidget(self.note)
        self.status=QtWidgets.QLabel('Load an approved dataset and its original models folder.');self.status.setWordWrap(True);layout.addWidget(self.status)
        self.draw()
    def angle_changed(self, value):
        self.slider.blockSignals(True);self.slider.setValue(round(value*10));self.slider.blockSignals(False);self.draw()
    def run_task(self, function, callback):
        if self.busy:return
        self.busy=True;self.controls.setEnabled(False);self.edit_controls.setEnabled(False)
        task=Task(function);self.task=task
        def finish(result):
            self.busy=False;self.applying=False;self.controls.setEnabled(True);self.edit_controls.setEnabled(True);self.task=None
            try:callback(result)
            except Exception as exc:self.error(str(exc))
        def failed(message):
            self.busy=False;self.applying=False;self.controls.setEnabled(True);self.edit_controls.setEnabled(True);self.task=None;self.error(message)
        task.signals.done.connect(finish);task.signals.failed.connect(failed);task.signals.progress.connect(self.status.setText)
        self.pool.start(task)
    def error(self, message):
        self.status.setText(message);QtWidgets.QMessageBox.warning(self,'BioCal3D Studio — Orientation',str(message))
    @staticmethod
    def sample_id(r):return r.get('sample_id',Path(r['sample_dir']).name)
    def confirm_discard(self):
        if not self.dirty:return True
        choice=QtWidgets.QMessageBox.question(self,'Unsaved rotations','Save your accepted rotations before continuing?',QtWidgets.QMessageBox.StandardButton.Save|QtWidgets.QMessageBox.StandardButton.Discard|QtWidgets.QMessageBox.StandardButton.Cancel)
        if choice==QtWidgets.QMessageBox.StandardButton.Cancel:return False
        if choice==QtWidgets.QMessageBox.StandardButton.Save:return self.save()
        self.dirty=False;return True
    def choose_dataset(self):
        folder=QtWidgets.QFileDialog.getExistingDirectory(self,'Full original Approved ROI snapshot',str(self.data_root or ''))
        if folder:self.load_dataset(Path(folder))
    def load_dataset(self, root):
        if self.busy or not self.confirm_discard():return False
        root=Path(root).resolve();self.status.setText('Reading inventory and frame mappings…')
        def work():
            if core.existing_stamp(root):raise ValueError('This dataset is already oriented. Choose the original approved snapshot for orientation.')
            rr=core.inventory(root)
            if not any(r['scanner']=='TRIOS5' for r in rr):raise ValueError('TRIOS5 is missing from this dataset.')
            return rr,core.frame_evidence(root,rr)
        def done(result):
            self.data_root=root;self.rr,self.evidence=result;self.cache={};self.reference=self.source=None;self.needs_fit=True
            self.config={'schema':'biocal3d_scanner_orientation_v1','config_id':uuid.uuid4().hex,'created_utc':core.now(),'reference':'TRIOS5',
                         'method':'manually reviewed rotation about existing canonical Z; rigid; no labels used','source_dataset':str(root),'frame_evidence':self.evidence,
                         'corrections':{'TRIOS5':{'angle_deg':0.,'matrix':np.eye(4).tolist(),'reviewed':True}}}
            present=[s for s in core.SCANNERS if any(r['scanner']==s for r in self.rr)]
            ref=sorted({self.sample_id(r) for r in self.rr if r['scanner']=='TRIOS5'})
            sets=[{self.sample_id(r) for r in self.rr if r['scanner']==s} for s in present]
            common=sorted(set.intersection(*sets));preferred=common[0] if common else ref[0]
            for box in (self.ref_sample,self.scanner,self.sample):box.blockSignals(True);box.clear()
            self.ref_sample.addItems(ref);self.ref_sample.setCurrentText(preferred);self.scanner.addItems([s for s in present if s!='TRIOS5'])
            for box in (self.ref_sample,self.scanner,self.sample):box.blockSignals(False)
            self.dirty=False;self.dataset_changed.emit(str(root));self.select_scanner();self.draw()
            if not self.raw:self.status.setText(f'{len(self.rr)} scans loaded. Choose the original full-model folder.')
        self.run_task(work,done);return True
    def set_original_root(self, folder):
        if self.busy:return
        self.raw=OriginalModelIndex(folder);self.cache={};self.load_previews()
    def choose_raw(self):
        folder=QtWidgets.QFileDialog.getExistingDirectory(self,'Original full-model data folder')
        if folder:self.set_original_root(folder);self.originals_changed.emit(str(Path(folder).resolve()))
    def select_scanner(self,*_):
        if self.busy:return
        values=sorted({self.sample_id(r) for r in self.rr if r['scanner']==self.scanner.currentText()})
        self.sample.blockSignals(True);self.sample.clear();self.sample.addItems(values)
        if self.ref_sample.currentText() in values:self.sample.setCurrentText(self.ref_sample.currentText())
        self.sample.blockSignals(False)
        self.angle.setValue((self.config or {}).get('corrections',{}).get(self.scanner.currentText(),{}).get('angle_deg',0));self.source=None;self.pending=[];self.needs_fit=True;self.draw();self.load_previews()
    def load_previews(self,*_):
        if self.busy or not self.rr or not self.raw or not self.scanner.currentText():return
        ref=next(r for r in self.rr if r['scanner']=='TRIOS5' and self.sample_id(r)==self.ref_sample.currentText())
        src=next(r for r in self.rr if r['scanner']==self.scanner.currentText() and self.sample_id(r)==self.sample.currentText())
        self.status.setText('Loading original models in their existing ROI coordinate frame…')
        def work():
            result=[]
            for r in (ref,src):
                key=r['scan_id']
                if key not in self.cache:
                    self.cache[key]=preview_model(self.data_root,r,self.raw)
                    while len(self.cache)>4:self.cache.pop(next(iter(self.cache)))
                result.append(self.cache[key])
            return result
        def done(result):
            self.reference,self.source=result;self.pending=[];self.needs_fit=True;self.draw()
            self.status.setText('Reference: '+self.reference['path']+' | Source: '+self.source['path'])
        self.run_task(work,done)
    def fit(self):
        self.views.fit([m for m in (self.reference,self.source) if m is not None])
    def draw(self):
        self.views.pick_mode=bool(self.pending)
        self.views.set_models(self.reference,self.source,self.angle.value(),self.scanner.currentText())
        if getattr(self,'needs_fit',False):self.fit();self.needs_fit=False
    def landmarks(self):
        if not self.source:return
        self.views.top();self.pending=['ref',[]];self.views.pick_mode=True
        self.note.setText('Top view restored. Click A then B on the LEFT TRIOS5 model, then the same ordered landmarks on the MIDDLE scanner model. Middle drag pans; scroll zooms.')
    def cancel_landmarks(self):
        self.pending=[];self.views.pick_mode=False;self.draw()
        self.note.setText('Landmark selection cancelled. Left drag rotates, middle drag pans, scroll zooms; all views remain linked.')
    def pick_landmark(self,index,point):
        if not self.pending:return
        phase,pts=self.pending
        if index!=(0 if phase=='ref' else 1):return
        pts.append(list(point[:2]));self.views.mark(index,point)
        if len(pts)<2:return
        if np.linalg.norm(np.subtract(pts[1],pts[0]))<1:
            self.pending=[];self.views.pick_mode=False;self.note.setText('Select landmarks at least 1 mm apart.');return
        if phase=='ref':self.ref_points=pts;self.pending=['src',[]];self.note.setText('Now click the SAME ordered pair A then B on the MIDDLE model.');return
        u=np.subtract(self.ref_points[1],self.ref_points[0]);v=np.subtract(pts[1],pts[0]);self.pending=[];self.views.pick_mode=False
        delta=math.degrees(math.atan2(u[1],u[0])-math.atan2(v[1],v[0]));self.angle.setValue((self.angle.value()+delta+180)%360-180);self.draw()
        self.note.setText('Landmark direction matched. Inspect the overlay, refine the angle, then Accept scanner.')
    def accept(self):
        if self.busy or not self.source or not self.config:return
        a=self.angle.value();scanner=self.scanner.currentText()
        self.config['corrections'][scanner]={'angle_deg':a,'matrix':core.rotation(a).tolist(),'reviewed':True,'accepted_utc':core.now(),
            'reference_model':{k:self.reference[k] for k in ('path','appearance','sample_id','source_sha256')},'source_model':{k:self.source[k] for k in ('path','appearance','sample_id','source_sha256')}}
        self.dirty=True;self.status.setText(f'Accepted {scanner}: {a:.2f}°. Save when all scanners are reviewed.')
    def save(self):
        if self.busy or not self.config:return False
        missing={r['scanner'] for r in self.rr}-set(self.config['corrections'])
        if missing:self.error('Accept rotations for: '+', '.join(sorted(missing)));return False
        path,_=QtWidgets.QFileDialog.getSaveFileName(self,'Save reusable rotations','scanner_rotations_TRIOS5.json','JSON (*.json)')
        if not path:return False
        try:
            core.dump(path,self.config);core.load_config(path);self.dirty=False
            try:self.views.capture(Path(path).with_suffix('.preview.png'));self.status.setText('Saved '+path)
            except Exception as exc:self.status.setText('Rotations saved: '+path+' | Preview unavailable: '+str(exc))
            return True
        except Exception as exc:self.error(str(exc));return False
    def load_saved(self):
        if not self.data_root:self.error('Load the full original approved dataset first.');return
        if not self.confirm_discard():return
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'Load saved rotations','','JSON (*.json)')
        if not path:return
        try:
            c=core.load_config(path);core.preflight(self.data_root,c);self.config=c;self.dirty=False
            self.angle.setValue(c['corrections'].get(self.scanner.currentText(),{}).get('angle_deg',0));self.status.setText('Loaded '+path)
        except Exception as exc:self.error(str(exc))
    def apply(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'Saved scanner rotations','','JSON (*.json)')
        if not path:return
        try:c=core.load_config(path)
        except Exception as exc:self.error(str(exc));return
        folder=QtWidgets.QFileDialog.getExistingDirectory(self,'Original dataset, calibration/agreement folder, or parent',str(self.data_root or ''))
        if not folder:return
        output,_=QtWidgets.QFileDialog.getSaveFileName(self,'Choose NEW output folder name',str(Path(folder).with_name(Path(folder).name+'_TRIOS5_oriented')),'Folder name (*)')
        if not output:return
        if not self.before_apply():return
        self.applying=True
        self.status.setText('Checking frames and labels before creating the oriented copy…')
        def work():return core.apply_tree(folder,output,copy.deepcopy(c),lambda message:self.task.signals.progress.emit(message))
        def done(out):self.status.setText('Oriented copy ready: '+str(out));self.output_ready.emit(str(out))
        self.run_task(work,done)

    def closeEvent(self,event):
        if self.busy:event.ignore();return
        if not self.confirm_discard():event.ignore();return
        self.views.dispose();event.accept()
