"""Interactive ROI verification; annotations are disabled while geometry is edited."""
from dataclasses import asdict
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from PySide6 import QtCore,QtWidgets
from roi_review_store import (load_review_context,save_review,current_version,read_state,Extraction,select_surface)

class ReviewDialog(QtWidgets.QDialog):
    def __init__(self,owner):
        super().__init__(owner)
        from biocal3d_annotator import SurfaceView
        self.owner=owner;self.setWindowTitle('Verify specimen surface • shared geometry');self.resize(1420,940)
        self.version=current_version(owner.store,owner.mesh.path);self.saved=False;self.actors=[]
        self.base,self.mask,self.settings,message,self.raw_path=load_review_context(owner.store,owner.mesh)
        self.initial=self.mask.copy();self.undo=[];self.redo=[];self.stroke=None
        layout=QtWidgets.QHBoxLayout(self)
        scroll=QtWidgets.QScrollArea();scroll.setWidgetResizable(True);scroll.setMaximumWidth(340)
        panel=QtWidgets.QWidget();left=QtWidgets.QVBoxLayout(panel);scroll.setWidget(panel);layout.addWidget(scroll)
        self.message=QtWidgets.QLabel(message);self.message.setWordWrap(True);left.addWidget(self.message)
        locate=QtWidgets.QPushButton('Locate original model…');locate.clicked.connect(self.locate);left.addWidget(locate)
        self.original=QtWidgets.QCheckBox('Show excluded / original model');self.original.setChecked(True)
        self.guides=QtWidgets.QCheckBox('Show circle, plane and height limits');self.guides.setChecked(True)
        for widget in (self.original,self.guides):left.addWidget(widget);widget.toggled.connect(self.render)
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
        for title,callback in [('Undo correction',self.undo_edit),('Redo correction',self.redo_edit),('Reset to opened surface',self.reset_mask),
                               ('Top view',self.top),('Side view',self.side)]:
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
        help_text=QtWidgets.QLabel('Green = retained surface • Red = excluded geometry • Space + left drag = rotate • Brush affects visible faces only')
        help_text.setWordWrap(True);right.addWidget(help_text)
        self.view=SurfaceView();right.addWidget(self.view,1)
        self.view.set_mesh(self.base);self.view.top_only=False
        self.view.stroke_started.connect(self.start_stroke);self.view.faces_painted.connect(self.paint_faces);self.view.stroke_finished.connect(self.end_stroke)
        self.radius.valueChanged.connect(lambda v:setattr(self.view,'radius',v));self.view.brush_changed.connect(self.radius.setValue)
        self.paint.toggled.connect(lambda enabled:setattr(self.view,'paint_mode',enabled))
        self.view.failure.connect(self.error)
        self.timer=QtCore.QTimer(self);self.timer.setSingleShot(True);self.timer.setInterval(180);self.timer.timeout.connect(self.preview)
        self.render();self.top()

    def error(self,message):QtWidgets.QMessageBox.warning(self,'Surface review',str(message))

    def read_settings(self):
        self.settings=Extraction(**{k:v.value() for k,v in self.controls.items()},connected=self.connected.isChecked())
        return self.settings

    def parameters_changed(self,*args):
        if not hasattr(self,'view'):return
        self.read_settings();self.render()
        if self.live.isChecked():self.timer.start()

    def push(self):self.undo.append(self.mask.copy());self.undo=self.undo[-30:];self.redo=[]
    def start_stroke(self):self.stroke=self.mask.copy()
    def paint_faces(self,ids):
        ids=np.asarray(ids,int)
        if not self.original.isChecked():ids=ids[self.mask[ids]]
        self.mask[ids]=self.operation.currentIndex()==1;self.render()
    def end_stroke(self):
        if self.stroke is not None and not np.array_equal(self.stroke,self.mask):
            self.undo.append(self.stroke);self.undo=self.undo[-30:];self.redo=[]
        self.stroke=None
    def undo_edit(self):
        if self.undo:self.redo.append(self.mask.copy());self.mask=self.undo.pop();self.render()
    def redo_edit(self):
        if self.redo:self.undo.append(self.mask.copy());self.mask=self.redo.pop();self.render()
    def reset_mask(self):self.push();self.mask=self.initial.copy();self.render()
    def automatic(self):
        try:
            mask,metrics=select_surface(self.base.vertices,self.base.faces,self.read_settings())
            if not mask.any():raise ValueError('No surface remains. Adjust the circle, plane or limits.')
            self.push();self.mask=mask;self.render()
        except Exception as e:self.error(e)
    def preview(self):self.automatic()

    def locate(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'Select the original scanner mesh',self.raw_path,'Mesh (*.ply *.vtp)')
        if not path:return
        try:
            base,mask,settings,message,raw_path=load_review_context(self.owner.store,self.owner.mesh,path)
            if base.path.resolve()!=__import__('pathlib').Path(path).resolve():
                raise ValueError('No verified original transform/triangle mapping was found. Local cleanup remains available.')
            self.base,self.mask,self.settings,self.raw_path=base,mask,settings,raw_path
            self.initial=mask.copy();self.undo=[];self.redo=[];self.message.setText(message)
            self.view.set_mesh(base);self.view.top_only=False;self.render();self.top()
        except Exception as e:self.error(e)

    def render(self,*args):
        if not hasattr(self,'view'):return
        # Ghost cells preserve original face indices while hiding excluded faces.
        hidden=np.zeros(len(self.mask),np.uint8)
        if not self.original.isChecked():hidden[~self.mask]=vtk.vtkDataSetAttributes.HIDDENCELL
        for data in (self.view.render_data,self.view.overlay_data):
            a=numpy_to_vtk(hidden,deep=True);a.SetName(vtk.vtkDataSetAttributes.GhostArrayName());data.GetCellData().AddArray(a);data.Modified()
        rgba=np.zeros((len(self.mask),4),np.uint8);rgba[self.mask]=[45,220,145,95];rgba[~self.mask]=[235,90,75,70]
        a=numpy_to_vtk(rgba,deep=True);a.SetName('ROIReview');self.view.overlay_data.GetCellData().SetScalars(a)
        for actor in self.actors:self.view.renderer.RemoveActor(actor)
        self.actors=[]
        if self.guides.isChecked():
            c,u,v,n=self.settings.frame();r=self.settings.radius
            for offset,color in [(0.,(1.,.85,.20)),(-self.settings.below,(.35,.7,1.)),(self.settings.above,(.35,.7,1.))]:
                circle=vtk.vtkRegularPolygonSource();circle.SetCenter(*(c+offset*n));circle.SetNormal(*n);circle.SetRadius(r);circle.SetNumberOfSides(160);circle.GeneratePolygonOff();circle.Update()
                mapper=vtk.vtkPolyDataMapper();mapper.SetInputData(circle.GetOutput());actor=vtk.vtkActor();actor.SetMapper(mapper)
                actor.GetProperty().SetColor(*color);actor.GetProperty().SetLineWidth(2);actor.PickableOff();self.view.renderer.AddActor(actor);self.actors.append(actor)
            plane=vtk.vtkPlaneSource();plane.SetOrigin(*(c-r*u-r*v));plane.SetPoint1(*(c+r*u-r*v));plane.SetPoint2(*(c-r*u+r*v));plane.Update()
            mapper=vtk.vtkPolyDataMapper();mapper.SetInputData(plane.GetOutput());actor=vtk.vtkActor();actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.,.85,.20);actor.GetProperty().SetOpacity(.10);actor.PickableOff();self.view.renderer.AddActor(actor);self.actors.append(actor)
        self.view.review_guides=self.actors
        self.count.setText(f'{int(self.mask.sum()):,} retained / {len(self.mask):,} source faces\nOpened revision: {self.version[:8] or "original crop"}')
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
            if pending:self.preview()
            if self.live.isChecked() and any(getattr(self.settings,k)!=v.value() for k,v in self.controls.items()):self.preview()
            save_review(self.owner.store,self.owner.mesh.path,self.base,self.mask,self.read_settings(),status,
                        self.notes.toPlainText(),self.version,self.raw_path)
            self.saved=True;self.accept()
        except Exception as e:self.error(e)

    def done(self,result):
        self.timer.stop()
        self.view.paint_mode=False
        super().done(result)
