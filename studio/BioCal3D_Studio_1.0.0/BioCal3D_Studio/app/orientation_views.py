"""Linked VTK orientation views using the same trackball controls as Inspect."""
import numpy as np
import vtk
from PySide6 import QtCore,QtWidgets
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import roi_browser_core as display


class OrientationViews(QtWidgets.QWidget):
    landmark=QtCore.Signal(int,object)
    def __init__(self,parent=None):
        super().__init__(parent);self.widgets=[];self.renderers=[];self.headers=[];self.actors=[]
        self.syncing=False;self.source=None;self.pick_mode=False;self.markers=[]
        self.sync_timer=QtCore.QTimer(self);self.sync_timer.setSingleShot(True);self.sync_timer.timeout.connect(self.flush)
        layout=QtWidgets.QHBoxLayout(self);layout.setContentsMargins(0,0,0,0)
        for index,title in enumerate(('TRIOS5 | fixed reference','Scanner | rotation','Aligned overlay')):
            panel=QtWidgets.QGroupBox();box=QtWidgets.QVBoxLayout(panel);layout.addWidget(panel,1)
            header=QtWidgets.QLabel(title);header.setWordWrap(True);header.setStyleSheet('font-size:15px;font-weight:600');box.addWidget(header);self.headers.append(header)
            widget=QVTKRenderWindowInteractor(panel);widget.setMinimumSize(200,220);box.addWidget(widget,1)
            renderer=vtk.vtkRenderer();renderer.SetBackground(.075,.095,.13);widget.GetRenderWindow().AddRenderer(renderer);widget.GetRenderWindow().SetMultiSamples(0)
            interactor=widget.GetRenderWindow().GetInteractor();interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
            interactor.AddObserver('CharEvent',lambda *_:None,1.)
            interactor.AddObserver('InteractionEvent',lambda *_,i=index:self.queue(i))
            interactor.AddObserver('EndInteractionEvent',lambda *_,i=index:self.queue(i))
            renderer.GetActiveCamera().AddObserver('ModifiedEvent',lambda *_,i=index:self.queue(i))
            widget.installEventFilter(self);widget.Initialize()
            self.widgets.append(widget);self.renderers.append(renderer)
        self.fit([])
    def queue(self,index):
        if self.syncing:return
        self.source=index;self.sync_timer.start(0)
    def flush(self):
        if self.source is None:return
        self.syncing=True
        try:
            camera=display.camera_state(self.renderers[self.source])
            for index,renderer in enumerate(self.renderers):
                if index!=self.source:display.set_camera(renderer,camera);self.widgets[index].GetRenderWindow().Render()
        finally:self.syncing=False
    def set_models(self,reference,source,angle,scanner):
        self.sync_timer.stop();self.syncing=True
        try:
            self.markers=[];self.actors=[]
            for renderer in self.renderers:renderer.RemoveAllViewProps()
            self.headers[1].setText(f'{scanner or "Scanner"} | {angle:.2f}°')
            for model,indices,rotation in ((reference,(0,2),0),(source,(1,2),angle)):
                if model is None:continue
                for index in indices:
                    actor=display.actor_for(model,'colour' if model['association']!='none' or model['texture'] else 'geometry',rotation)
                    if index==2:actor.GetProperty().SetOpacity(.65)
                    self.renderers[index].AddActor(actor);self.actors.append(actor)
            for renderer,widget in zip(self.renderers,self.widgets):renderer.ResetCameraClippingRange();widget.GetRenderWindow().Render()
        finally:self.syncing=False
    def fit(self,models):
        self.sync_timer.stop();self.syncing=True
        try:
            camera=display.default_camera(models)
            for renderer,widget in zip(self.renderers,self.widgets):display.set_camera(renderer,camera);widget.GetRenderWindow().Render()
        finally:self.syncing=False
    def top(self):
        self.sync_timer.stop();self.syncing=True
        try:
            c=self.renderers[0].GetActiveCamera();focus=c.GetFocalPoint();distance=max(c.GetDistance(),1.)
            camera={'position':[focus[0],focus[1],focus[2]+distance],'focal_point':list(focus),'view_up':[0,1,0],'parallel_scale':c.GetParallelScale()}
            for renderer,widget in zip(self.renderers,self.widgets):display.set_camera(renderer,camera);widget.GetRenderWindow().Render()
        finally:self.syncing=False
    def eventFilter(self,obj,event):
        if self.pick_mode and obj in self.widgets and event.type() in (QtCore.QEvent.Type.MouseButtonPress,QtCore.QEvent.Type.MouseButtonRelease,QtCore.QEvent.Type.MouseButtonDblClick) and event.button()==QtCore.Qt.MouseButton.LeftButton:
            if event.type()==QtCore.QEvent.Type.MouseButtonPress:
                index=self.widgets.index(obj);picker=vtk.vtkCellPicker();picker.SetTolerance(.005)
                scale=obj.devicePixelRatioF();position=event.position()
                if picker.Pick(position.x()*scale,(obj.height()-position.y()-1)*scale,0,self.renderers[index]):
                    self.landmark.emit(index,np.asarray(picker.GetPickPosition(),float))
            return True
        return super().eventFilter(obj,event)
    def mark(self,index,point):
        source=vtk.vtkSphereSource();source.SetCenter(*point);source.SetRadius(max(self.renderers[index].GetActiveCamera().GetParallelScale()*.012,.03))
        mapper=vtk.vtkPolyDataMapper();mapper.SetInputConnection(source.GetOutputPort());actor=vtk.vtkActor();actor.SetMapper(mapper);actor.GetProperty().SetColor(1.,.32,.48);actor.GetProperty().LightingOff();actor.PickableOff()
        self.renderers[index].AddActor(actor);self.markers.append(actor);self.widgets[index].GetRenderWindow().Render()
    def capture(self,path):
        from PIL import Image,ImageDraw
        from vtk.util.numpy_support import vtk_to_numpy
        images=[]
        for widget in self.widgets:
            window=widget.GetRenderWindow();window.Render()
            capture=vtk.vtkWindowToImageFilter();capture.SetInput(window);capture.SetInputBufferTypeToRGB();capture.ReadFrontBufferOff();capture.Update()
            data=capture.GetOutput();w,h,_=data.GetDimensions();rgb=vtk_to_numpy(data.GetPointData().GetScalars()).reshape(h,w,3)
            images.append(Image.fromarray(np.flipud(rgb).copy()))
        result=Image.new('RGB',(sum(im.width for im in images),max(im.height for im in images)+30),'#111923');draw=ImageDraw.Draw(result);x=0
        for image,header in zip(images,self.headers):result.paste(image,(x,30));draw.text((x+8,8),header.text(),fill='#dfe8f5');x+=image.width
        result.save(path)
    def dispose(self):
        self.sync_timer.stop()
        for widget in self.widgets:widget.Finalize()
