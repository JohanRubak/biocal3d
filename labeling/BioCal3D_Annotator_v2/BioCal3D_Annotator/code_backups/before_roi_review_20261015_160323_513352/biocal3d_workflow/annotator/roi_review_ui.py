"""Interactive ROI verification; annotations are disabled while geometry is edited."""
from dataclasses import asdict
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from PySide6 import QtCore,QtGui,QtWidgets
from roi_review_store import (load_review_context,save_review,current_version,read_state,Extraction,select_surface)

class ReviewDialog(QtWidgets.QDialog):
    def __init__(self,owner):
        super().__init__(owner)
        from biocal3d_annotator import SurfaceView
        self.owner=owner;self.setWindowTitle('Verify specimen surface • shared geometry')
        self.setWindowFlags(QtCore.Qt.WindowType.Window|QtCore.Qt.WindowType.WindowTitleHint|
                            QtCore.Qt.WindowType.WindowSystemMenuHint|QtCore.Qt.WindowType.WindowMinimizeButtonHint|
                            QtCore.Qt.WindowType.WindowMaximizeButtonHint|QtCore.Qt.WindowType.WindowCloseButtonHint)
        self.setSizeGripEnabled(True)
        screen=owner.screen() or QtWidgets.QApplication.primaryScreen()
        available=screen.availableGeometry() if screen else QtCore.QRect(0,0,1600,1000)
        self.resize(int(available.width()*.95),int(available.height()*.92))
        self.setMinimumSize(min(1080,int(available.width()*.9)),min(700,int(available.height()*.85)))
        self.setWindowState(self.windowState()|QtCore.Qt.WindowState.WindowMaximized)
        self._visible_request=None;self.preview_metrics=None
        self.version=current_version(owner.store,owner.mesh.path);self.saved=False;self.actors=[]
        self.base,self.mask,self.settings,message,self.raw_path=load_review_context(owner.store,owner.mesh)
        self.initial=self.mask.copy();self.undo=[];self.redo=[];self.stroke=None
        layout=QtWidgets.QHBoxLayout(self)
        scroll=QtWidgets.QScrollArea();scroll.setWidgetResizable(True);scroll.setMinimumWidth(340);scroll.setMaximumWidth(390);scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        panel=QtWidgets.QWidget();left=QtWidgets.QVBoxLayout(panel);scroll.setWidget(panel);layout.addWidget(scroll)
        self.message=QtWidgets.QLabel(message);self.message.setWordWrap(True);left.addWidget(self.message)
        locate=QtWidgets.QPushButton('Locate original model…');locate.clicked.connect(self.locate);left.addWidget(locate)
        self.original=QtWidgets.QCheckBox('Show excluded / original model');self.original.setChecked(False)
        self.guides=QtWidgets.QCheckBox('Show circle, plane and height limits');self.guides.setChecked(True)
        for widget in (self.original,self.guides):left.addWidget(widget)
        self.original.toggled.connect(self.context_toggled);self.guides.toggled.connect(self.render)
        left.addWidget(QtWidgets.QLabel('Extraction parameters (mm / degrees)'))
        form=QtWidgets.QFormLayout();self.controls={}
        for name,label,lo,hi,step in [('cx','Centre X',-10000,10000,.1),('cy','Centre Y',-10000,10000,.1),
            ('z','Plane height',-10000,10000,.1),('radius','Radius',.01,10000,.1),
            ('tilt_x','Plane tilt X',-80,80,1),('tilt_y','Plane tilt Y',-80,80,1),
            ('below','Depth below plane',0,10000,.1),('above','Height above plane',0,10000,.1),
            ('max_angle','Maximum slope',1,90,1)]:
            box=QtWidgets.QDoubleSpinBox();box.setRange(lo,hi);box.setDecimals(3);box.setSingleStep(step)
            box.setValue(getattr(self.settings,name));box.valueChanged.connect(self.parameters_changed)
            self.controls[name]=box;form.addRow(label,box)
        left.addLayout(form)
        self.connected=QtWidgets.QCheckBox('Keep central connected surface');self.connected.setChecked(self.settings.connected)
        self.connected.toggled.connect(self.parameters_changed);left.addWidget(self.connected)
        self.live=QtWidgets.QCheckBox('Live extraction preview');left.addWidget(self.live)
        self.live.toggled.connect(lambda enabled:(self.timer.start() if enabled else self.timer.stop()) if hasattr(self,'timer') else None)
        auto=QtWidgets.QPushButton('Apply automatic selection');auto.clicked.connect(self.automatic);left.addWidget(auto)
        note=QtWidgets.QLabel('Automatic selection replaces manual edits. Plane/angle limits select faces; they never flatten the mesh.')
        note.setWordWrap(True);left.addWidget(note)
        left.addWidget(QtWidgets.QLabel('Manual surface correction'))
        self.operation=QtWidgets.QComboBox();self.operation.addItems(['Exclude surface','Restore surface'])
        left.addWidget(self.operation)
        self.paint=QtWidgets.QCheckBox('Brush enabled');self.paint.setChecked(True);left.addWidget(self.paint)
        self.radius=QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal);self.radius.setRange(3,150);self.radius.setValue(24);left.addWidget(self.radius)
        for title,callback in [('Undo correction [Ctrl+Z]',self.undo_edit),('Redo correction [Ctrl+Y]',self.redo_edit),('Reset to opened surface',self.reset_mask),
                               ('Top view [R]',self.top),('Side view',self.side),('Fit visible model [F]',self.fit_visible)]:
            b=QtWidgets.QPushButton(title);b.clicked.connect(callback);left.addWidget(b)
        self.count=QtWidgets.QLabel();self.count.setWordWrap(True);left.addWidget(self.count)
        self.notes=QtWidgets.QPlainTextEdit();self.notes.setPlaceholderText('Geometry review notes / reason for flagging');self.notes.setMaximumHeight(90)
        state=read_state(owner.store,owner.mesh.path)
        if state:self.notes.setPlainText(state.get('notes',''))
        left.addWidget(self.notes)
        for title,status in [('Save pending review','pending'),('Approve shared surface','approved'),('Flag for investigation','flagged')]:
            b=QtWidgets.QPushButton(title);b.clicked.connect(lambda checked=False,s=status:self.commit(s));left.addWidget(b)
        close=QtWidgets.QPushButton('Cancel');close.clicked.connect(self.reject);left.addWidget(close);left.addStretch()
        right=QtWidgets.QVBoxLayout();layout.addLayout(right,1)
        help_text=QtWidgets.QLabel('Green = retained • Red = excluded context (toggle off to see crop only) • Space + left drag = rotate')
        help_text.setWordWrap(True);right.addWidget(help_text)
        self.preview_status=QtWidgets.QLabel('Adjust parameters and apply selection, or enable live preview.');self.preview_status.setWordWrap(True);right.addWidget(self.preview_status)
        self.view=SurfaceView();right.addWidget(self.view,1)
        self.view.set_mesh(self.base);self.view.top_only=False
        self.view.stroke_started.connect(self.start_stroke);self.view.faces_painted.connect(self.paint_faces);self.view.stroke_finished.connect(self.end_stroke)
        self.radius.valueChanged.connect(lambda v:setattr(self.view,'radius',v));self.view.brush_changed.connect(self.radius.setValue)
        self.paint.toggled.connect(lambda enabled:setattr(self.view,'paint_mode',enabled))
        self.view.failure.connect(self.error)
        self.timer=QtCore.QTimer(self);self.timer.setSingleShot(True);self.timer.setInterval(180);self.timer.timeout.connect(self.preview)
        self._applied_settings=asdict(self.settings)
        self.initial_state=self.snapshot()
        self.shortcuts={}
        for key,callback in [('Ctrl+Z',self.undo_edit),('Ctrl+Y',self.redo_edit),('Ctrl+Shift+Z',self.redo_edit),
                             ('F',self.fit_visible),('R',self.top)]:
            shortcut=QtGui.QShortcut(QtGui.QKeySequence(key),self)
            shortcut.setContext(QtCore.Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(callback);self.shortcuts[key]=shortcut
        QtWidgets.QApplication.instance().focusChanged.connect(self.update_shortcuts)
        QtWidgets.QApplication.instance().installEventFilter(self)
        self.update_shortcuts();self.render();self.top()

    def error(self,message):QtWidgets.QMessageBox.warning(self,'Surface review',str(message))

    def read_settings(self):
        self.settings=Extraction(**{k:v.value() for k,v in self.controls.items()},connected=self.connected.isChecked())
        return self.settings

    def parameters_changed(self,*args):
        if not hasattr(self,'view'):return
        self.read_settings();self.render()
        if self.live.isChecked():self.timer.start()

    def editing_context(self):
        focus=QtWidgets.QApplication.focusWidget()
        spin=isinstance(focus,QtWidgets.QAbstractSpinBox) or (focus is not None and isinstance(focus.parentWidget(),QtWidgets.QAbstractSpinBox))
        text=isinstance(focus,(QtWidgets.QLineEdit,QtWidgets.QPlainTextEdit,QtWidgets.QTextEdit)) and not spin
        return text,spin

    def update_shortcuts(self,*args):
        text,spin=self.editing_context()
        for key,shortcut in self.shortcuts.items():
            shortcut.setEnabled(not (text if key.startswith('Ctrl+') else text or spin))

    def eventFilter(self,obj,event):
        kind=event.type()
        if (kind in (QtCore.QEvent.Type.ShortcutOverride,QtCore.QEvent.Type.KeyPress,QtCore.QEvent.Type.KeyRelease) and
            isinstance(obj,QtWidgets.QWidget) and obj.window() is self):
            modifiers=event.modifiers();key=event.key()
            ctrl=bool(modifiers&QtCore.Qt.KeyboardModifier.ControlModifier)
            other=bool(modifiers&(QtCore.Qt.KeyboardModifier.AltModifier|QtCore.Qt.KeyboardModifier.MetaModifier))
            shift=bool(modifiers&QtCore.Qt.KeyboardModifier.ShiftModifier)
            undo_key=ctrl and not other and key==QtCore.Qt.Key.Key_Z
            redo_key=ctrl and not other and key==QtCore.Qt.Key.Key_Y and not shift
            if (undo_key or redo_key) and not self.editing_context()[0]:
                # Numeric editors accept native Ctrl+Z before QShortcut runs.
                # Route these commands to geometry history exactly once.
                if kind==QtCore.QEvent.Type.ShortcutOverride:event.accept();return True
                if kind==QtCore.QEvent.Type.KeyPress:
                    if redo_key or shift:self.redo_edit()
                    else:self.undo_edit()
                return True
        return False

    def snapshot(self,settings=None):
        return dict(mask=self.mask.copy(),settings=dict(settings or asdict(self.settings)),metrics=self.preview_metrics,
                    summary=self.preview_status.text())

    def restore_state(self,state):
        self.timer.stop();self.mask=state['mask'].copy();self.settings=Extraction(**state['settings'])
        self._applied_settings=dict(state['settings']);self.preview_metrics=state['metrics']
        for name,control in self.controls.items():
            blocker=QtCore.QSignalBlocker(control);control.setValue(getattr(self.settings,name));del blocker
        blocker=QtCore.QSignalBlocker(self.connected);self.connected.setChecked(self.settings.connected);del blocker
        self.preview_status.setText(state['summary']);self.render()

    def push(self,state=None):
        self.undo.append(state if state is not None else self.snapshot());self.undo=self.undo[-30:];self.redo=[]

    def start_stroke(self):
        if self.timer.isActive():self.automatic()
        self.stroke=self.snapshot()

    def paint_faces(self,ids):
        ids=np.asarray(ids,int)
        if not self.original.isChecked():ids=ids[self.mask[ids]]
        self.mask[ids]=self.operation.currentIndex()==1;self.render()

    def end_stroke(self):
        if self.stroke is not None and not np.array_equal(self.stroke['mask'],self.mask):
            self.push(self.stroke);self._applied_settings=asdict(self.settings)
        self.stroke=None

    def finish_edit(self):
        self.view.painting=False;self.view.previous=None;self.end_stroke()
        if self.timer.isActive():self.automatic()
        self.timer.stop()

    def undo_edit(self):
        self.finish_edit()
        if self.undo:
            self.redo.append(self.snapshot());self.restore_state(self.undo.pop())

    def redo_edit(self):
        self.finish_edit()
        if self.redo:
            self.undo.append(self.snapshot());self.restore_state(self.redo.pop())

    def reset_mask(self):
        self.finish_edit();self.push();self.restore_state(self.initial_state)

    def automatic(self):
        try:
            before=self.snapshot(settings=self._applied_settings)
            mask,metrics=select_surface(self.base.vertices,self.base.faces,self.read_settings())
            self.timer.stop()
            changed=not np.array_equal(mask,self.mask)
            if changed or asdict(self.settings)!=self._applied_settings:self.push(before)
            self.mask=mask;self.preview_metrics=metrics;self._applied_settings=asdict(self.settings)
            summary=(f'Preview updated: below {self.settings.below:g} mm / above {self.settings.above:g} mm • '
                     f'{metrics["candidate_faces"]:,} faces within circle + height • '
                     f'{metrics["oriented_faces"]:,} after slope filter • {int(mask.sum()):,} retained.')
            if not mask.any():summary+=' Empty extraction: adjust the plane/limits before saving.'
            elif not changed:summary+=' Retained faces unchanged; no additional connected surface passes the current limits.'
            if not getattr(self.base,'context_kind','local')=='full':summary+=' Working within the existing crop; the full original is unavailable.'
            self.preview_status.setText(summary);self.render();return True
        except Exception as e:self.error(e);return False
    def preview(self):return self.automatic()

    def locate(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'Select the original scanner mesh',self.raw_path,'Mesh (*.ply *.vtp)')
        if not path:return
        try:
            base,mask,settings,message,raw_path=load_review_context(self.owner.store,self.owner.mesh,path)
            if base.path.resolve()!=__import__('pathlib').Path(path).resolve():
                raise ValueError('No verified original transform/triangle mapping was found. Local cleanup remains available.')
            self.base,self.mask,self.settings,self.raw_path=base,mask,settings,raw_path
            self.initial=mask.copy();self.undo=[];self.redo=[];self.message.setText(message)
            for name,control in self.controls.items():
                blocker=QtCore.QSignalBlocker(control);control.setValue(getattr(settings,name));del blocker
            blocker=QtCore.QSignalBlocker(self.connected);self.connected.setChecked(settings.connected);del blocker
            self._applied_settings=asdict(settings)
            self.view.set_mesh(base);self.view.top_only=False;self._visible_request=None;self.preview_metrics=None
            self.preview_status.setText('Original model loaded. Adjust parameters and apply selection, or enable live preview.')
            self.initial_state=self.snapshot();self.render();self.top()
        except Exception as e:self.error(e)

    def context_toggled(self,checked):
        # Visibility changes must preserve position, direction and zoom.
        # Reframing remains an explicit Fit visible model action.
        self.render()

    def fit_visible(self):
        self.view.renderer.ResetCamera();self.view.renderer.ResetCameraClippingRange();self.view.GetRenderWindow().Render()

    def render(self,*args):
        if not hasattr(self,'view'):return
        full=getattr(self.base,'context_kind','local')=='full'
        has_excluded=bool(np.any(~self.mask))
        self.original.setText('Show excluded / original model' if full else 'Show excluded ROI geometry (local only)')
        self.original.setEnabled(full or has_excluded)
        self.original.setToolTip('Show the complete verified scanner mesh.' if full else
                                'Full original model unavailable. Locate it using its saved geometry mapping. Only geometry present in this loaded crop can be shown.')
        requested=np.arange(len(self.mask),dtype=np.int64) if self.original.isChecked() else np.flatnonzero(self.mask)
        if self._visible_request is None or not np.array_equal(self._visible_request,requested):
            self.view.set_display_faces(requested);self._visible_request=requested.copy()
        ids=self.view.display_face_ids
        rgba=np.zeros((len(ids),4),np.uint8);retained=self.mask[ids]
        rgba[retained]=[45,220,145,95];rgba[~retained]=[235,90,75,110]
        a=numpy_to_vtk(rgba,deep=True);a.SetName('ROIReview');self.view.overlay_data.GetCellData().SetScalars(a)
        self.view.overlay_data.Modified();self.view.overlay_mapper.Modified();self.view.mapper.Modified()
        for actor in self.actors:self.view.renderer.RemoveActor(actor)
        self.actors=[]
        if self.guides.isChecked():
            c,u,v,n=self.settings.frame();r=self.settings.radius
            for offset,color in [(0.,(1.,.85,.20)),(-self.settings.below,(.35,.7,1.)),(self.settings.above,(.35,.7,1.))]:
                centre=c+offset*n
                circle=vtk.vtkRegularPolygonSource();circle.SetCenter(*centre);circle.SetNormal(*n);circle.SetRadius(r);circle.SetNumberOfSides(160);circle.GeneratePolygonOff();circle.Update()
                mapper=vtk.vtkPolyDataMapper();mapper.SetInputData(circle.GetOutput());actor=vtk.vtkActor();actor.SetMapper(mapper)
                actor.GetProperty().SetColor(*color);actor.GetProperty().SetLineWidth(2);actor.PickableOff();self.view.renderer.AddActor(actor);self.actors.append(actor)
                plane=vtk.vtkPlaneSource();plane.SetOrigin(*(centre-r*u-r*v));plane.SetPoint1(*(centre+r*u-r*v));plane.SetPoint2(*(centre-r*u+r*v));plane.Update()
                mapper=vtk.vtkPolyDataMapper();mapper.SetInputData(plane.GetOutput());actor=vtk.vtkActor();actor.SetMapper(mapper)
                actor.GetProperty().SetColor(*color);actor.GetProperty().SetOpacity(.10 if offset==0 else .05);actor.PickableOff();self.view.renderer.AddActor(actor);self.actors.append(actor)
        self.view.review_guides=self.actors
        mode='Full original loaded' if full else 'LOCAL CROP ONLY — locate original to show missing geometry'
        self.count.setText(f'{int(self.mask.sum()):,} retained / {len(self.mask):,} source faces\n{mode}\nOpened revision: {self.version[:8] or "original crop"}')
        self.view.projection_key=None;self.view.renderer.ResetCameraClippingRange();self.view.GetRenderWindow().Render()

    def top(self):self.camera(False)
    def side(self):self.camera(True)
    def camera(self,side):
        c,u,v,n=self.settings.frame();camera=self.view.renderer.GetActiveCamera()
        camera.SetFocalPoint(*c);camera.SetPosition(*(c+(v if side else n)*self.settings.radius*4));camera.SetViewUp(*(n if side else v))
        camera.ParallelProjectionOn();camera.SetParallelScale(self.settings.radius*1.35);self.view.renderer.ResetCameraClippingRange();self.view.GetRenderWindow().Render()

    def commit(self,status):
        try:
            self.end_stroke()
            pending=self.timer.isActive();self.timer.stop()
            if pending and not self.preview():return
            if self.live.isChecked() and any(getattr(self.settings,k)!=v.value() for k,v in self.controls.items()):
                if not self.preview():return
            if not self.mask.any():raise ValueError('The extraction is empty. Adjust the plane or limits before saving.')
            save_review(self.owner.store,self.owner.mesh.path,self.base,self.mask,self.read_settings(),status,
                        self.notes.toPlainText(),self.version,self.raw_path)
            self.saved=True;self.accept()
        except Exception as e:self.error(e)

    def done(self,result):
        app=QtWidgets.QApplication.instance();app.removeEventFilter(self)
        try:app.focusChanged.disconnect(self.update_shortcuts)
        except (RuntimeError,TypeError):pass
        self.timer.stop()
        self.view.paint_mode=False
        super().done(result)
