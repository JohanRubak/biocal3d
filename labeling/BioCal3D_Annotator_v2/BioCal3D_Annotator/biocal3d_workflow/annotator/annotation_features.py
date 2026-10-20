"""Display-only appearance controls, orientation cues and rater switching."""
import re
import csv
import json
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from PySide6 import QtCore, QtWidgets
if __package__:
    from .mesh_io import AnnotationStore, atomic_npz, surface_normals
else:
    from mesh_io import AnnotationStore, atomic_npz, surface_normals


def adjust_rgb(colors, brightness=0., contrast=1., gamma=1., saturation=1.):
    original=np.asarray(colors,dtype=np.uint8)
    if brightness==0 and contrast==1 and gamma==1 and saturation==1:return original.copy()
    rgb=original[...,:3].astype(np.float32)/255
    rgb=np.clip((rgb-.5)*contrast+.5+brightness,0,1)**(1/max(gamma,.01))
    gray=(rgb*np.array([.2126,.7152,.0722],np.float32)).sum(axis=-1,keepdims=True)
    out=original.copy();out[...,:3]=np.rint(np.clip(gray+(rgb-gray)*saturation,0,1)*255).astype(np.uint8)
    return out


def sample_frame(mesh):
    v=mesh.vertices[np.unique(mesh.faces[mesh.active])];center=v.mean(axis=0)
    _,_,axes=np.linalg.svd(v-center,full_matrices=False);normal=axes[-1].copy()
    canonical=False;geometry=mesh.path.with_name('geometry.npz')
    if geometry.exists():
        with np.load(geometry,allow_pickle=False) as d:canonical='original_to_canonical' in d
    if canonical:normal=np.array([0.,0.,1.])
    else:
        _,normals=surface_normals(mesh.vertices,mesh.faces)
        if np.dot(normal,normals[mesh.active].sum(axis=0))<0:normal=-normal
    up=axes[1]-np.dot(axes[1],normal)*normal
    if np.linalg.norm(up)<1e-8:up=np.array([0.,1.,0.])
    return center,normal,up/np.linalg.norm(up),'canonical reference +Z' if canonical else 'inferred mesh direction'


def orientation_path(store,mesh):
    rel=mesh.path.relative_to(store.root)
    return store.output/'_sample_orientation'/rel.parent/(rel.name+'.orientation.npz')


def read_orientation(store,mesh):
    path=orientation_path(store,mesh)
    if not path.exists():return None
    with np.load(path,allow_pickle=False) as d:
        if str(d['fingerprint'].item())!=mesh.fingerprint:
            # Reviewed surfaces retain the original coordinate frame, but an old
            # orientation fingerprint may belong to the immutable input topology.
            from mesh_io import load_mesh
            if not mesh.review_version or str(d['fingerprint'].item())!=load_mesh(mesh.path).fingerprint:
                if mesh.review_version:return None
                raise ValueError('Saved top direction does not match this geometry.')
        normal=np.array(d['top_normal'],float)
    if normal.shape!=(3,) or not np.isfinite(normal).all() or np.linalg.norm(normal)<1e-8:raise ValueError('Invalid saved top direction.')
    return normal/np.linalg.norm(normal)


class SurfaceFeatures:
    def init_features(self):
        self.color_adjustments=dict(brightness=0.,contrast=1.,gamma=1.,saturation=1.)
        self.color_enabled=True
        self.neutral_texture=None
        self.original_colors=None;self.original_texture=None;self.texture_pixels=None
        self.top_only=True;self.top_normal=np.array([0.,0.,1.]);self.surface_center=np.zeros(3);self.surface_up=np.array([0.,1.,0.])
        self.side_text=vtk.vtkTextActor();self.side_text.SetPosition(14,14);self.side_text.PickableOff()
        prop=self.side_text.GetTextProperty();prop.SetFontSize(18);prop.BoldOn();prop.SetBackgroundOpacity(.85)
        # AddViewProp works for text/2D actors across VTK versions, including
        # builds that no longer expose the legacy AddActor2D alias.
        self.renderer.AddViewProp(self.side_text)
        self.axes=vtk.vtkAxesActor();self.axes.SetZAxisLabelText('TOP')
        self.orientation_widget=vtk.vtkOrientationMarkerWidget();self.orientation_widget.SetOrientationMarker(self.axes)
        self.orientation_widget.SetInteractor(self.GetRenderWindow().GetInteractor());self.orientation_widget.SetViewport(0,.12,.16,.35)
        self.orientation_widget.SetEnabled(1);self.orientation_widget.InteractiveOff()
        self.renderer.GetActiveCamera().AddObserver('ModifiedEvent',lambda *_:self.update_side_indicator())

    def setup_orientation(self,mesh):
        self.surface_center,self.top_normal,self.surface_up,self.top_source=sample_frame(mesh)
        self.set_top_normal(self.top_normal)

    def cache_colors(self):
        a=self.mesh.display.GetPointData().GetScalars()
        self.original_colors=vtk_to_numpy(a).copy() if a is not None and a.GetNumberOfComponents()>=3 else None
        self.original_texture=None;self.texture_pixels=None

    def apply_point_style(self):
        # Point visibility is independent of scanner RGB, UV textures and sliders.
        self.points_mapper.ScalarVisibilityOff()
        self.points.SetTexture(None)
        prop = self.points.GetProperty()
        prop.SetColor(1., 1., 0.)
        prop.LightingOff()
        prop.SetRepresentationToPoints()

    def apply_color_adjustments(self):
        if self.mesh is None:return
        if not self.color_enabled:
            # Keep UV meshes on the textured rendering path. Removing a texture
            # can leave unlit geometry blank on some VTK/OpenGL builds until a
            # shading change rebuilds the shader. An opaque white texture gives
            # the actor's neutral gray color without showing scanner imagery.
            if self.original_texture is not None:
                if self.neutral_texture is None:
                    image=vtk.vtkImageData();image.SetDimensions(1,1,1)
                    image.GetPointData().SetScalars(numpy_to_vtk(np.full((1,3),255,np.uint8),deep=True))
                    self.neutral_texture=vtk.vtkTexture();self.neutral_texture.SetInputData(image)
                self.base.SetTexture(self.neutral_texture)
            else:
                self.base.SetTexture(None)
            self.mapper.ScalarVisibilityOff()
            self.base.GetProperty().SetColor(.8,.81,.82)
        elif self.original_texture is not None:
            image=vtk.vtkImageData();image.DeepCopy(self.original_texture)
            image.GetPointData().SetScalars(numpy_to_vtk(adjust_rgb(self.texture_pixels,**self.color_adjustments),deep=True))
            texture=vtk.vtkTexture();texture.SetInputData(image);texture.InterpolateOn()
            self.base.SetTexture(texture)
            self.mapper.ScalarVisibilityOff();self.adjusted_texture=image
        elif self.original_colors is not None:
            a=numpy_to_vtk(adjust_rgb(self.original_colors,**self.color_adjustments),deep=True);a.SetName('RGB')
            self.render_data.GetPointData().SetScalars(a)
            self.mapper.ScalarVisibilityOn()
        self.apply_point_style()
        for actor,mapper in ((self.base,self.mapper),(self.points,self.points_mapper)):
            actor.GetProperty().Modified();actor.Modified();mapper.Modified()
        self.GetRenderWindow().Render()

    def load_display_texture(self,path):
        if self.mesh is None or self.mesh.display.GetPointData().GetTCoords() is None:raise ValueError('Mesh has no UV coordinates.')
        reader=vtk.vtkImageReader2Factory().CreateImageReader2(str(path))
        if reader is None:raise ValueError('Cannot read this texture.')
        reader.SetFileName(str(path));reader.Update()
        if reader.GetOutput().GetNumberOfPoints()==0:raise ValueError('Texture is empty or unreadable.')
        self.original_texture=vtk.vtkImageData();self.original_texture.DeepCopy(reader.GetOutput())
        a=vtk_to_numpy(self.original_texture.GetPointData().GetScalars()).copy()
        if a.ndim==1:a=a[:,None]
        if a.shape[1]<3:a=np.repeat(a[:,:1],3,axis=1)
        self.texture_pixels=a.astype(np.uint8)
        self.mesh.texture=Path(path).resolve();self.mesh.warning='';self.apply_color_adjustments()

    def set_top_normal(self,normal):
        self.top_normal=np.asarray(normal,float)/np.linalg.norm(normal)
        up=self.surface_up-np.dot(self.surface_up,self.top_normal)*self.top_normal
        if np.linalg.norm(up)<1e-8:
            up=np.array([1.,0.,0.]) if abs(self.top_normal[0])<.9 else np.array([0.,1.,0.])
            up-=np.dot(up,self.top_normal)*self.top_normal
        self.surface_up=up/np.linalg.norm(up)
        basis=np.column_stack((np.cross(self.surface_up,self.top_normal),self.surface_up,self.top_normal))
        matrix=vtk.vtkMatrix4x4()
        for i in range(3):
            for j in range(3):matrix.SetElement(i,j,basis[i,j])
        self.axes.SetUserMatrix(matrix);self.update_side_indicator()

    def side_alignment(self):
        direction=np.asarray(self.renderer.GetActiveCamera().GetPosition())-self.surface_center
        return float(np.dot(direction,self.top_normal)/max(np.linalg.norm(direction),1e-12))

    def update_side_indicator(self):
        if self.mesh is None:self.side_text.SetInput('');return
        side=self.side_alignment()
        if side>.08:text='TOP - plaque surface';color=(.3,.95,.72);background=(.035,.12,.10)
        elif side<-.08:
            text='BOTTOM / BACKSIDE\n'+('Painting locked - R returns to top' if self.top_only else 'Top-only guard disabled')
            color=(1.,.48,.35);background=(.20,.045,.035)
        else:
            text='SIDE VIEW\n'+('Painting locked - rotate toward top' if self.top_only else 'Top-only guard disabled')
            color=(1.,.8,.3);background=(.16,.11,.025)
        self.side_text.SetInput(text);prop=self.side_text.GetTextProperty();prop.SetColor(*color);prop.SetBackgroundColor(*background)

    def reset_top_camera(self):
        if self.mesh is None:return
        v=self.mesh.vertices[np.unique(self.mesh.faces[self.mesh.active])]
        extent=max(float(np.linalg.norm(np.ptp(v,axis=0))),.01);camera=self.renderer.GetActiveCamera()
        camera.SetFocalPoint(*self.surface_center);camera.SetPosition(*(self.surface_center+self.top_normal*extent*2))
        camera.SetViewUp(*self.surface_up);camera.ParallelProjectionOn();camera.SetParallelScale(extent/2.4)
        self.renderer.ResetCameraClippingRange();self.GetRenderWindow().Render()


class AnnotatorFeatures:
    def queue_metadata(self,root,queue,manifest=None):
        metadata={}
        for path in queue:
            info={};source=path.with_name('metadata.json')
            if source.is_file():
                try:
                    value=json.loads(source.read_text(encoding='utf-8'))
                    if isinstance(value,dict):info=value
                except (OSError,ValueError):pass
            sample=path.parent.name if path.name.lower() in ('roi.ply','roi.vtp','geometry.npz') else path.stem
            match=re.search(r'BCG(\d+)T(\d+)[-_](\d+)',sample,re.I)
            if match:
                info.setdefault('group',match[1]);info.setdefault('physical_id',f'G{match[1]}_S{match[3]}')
            info.setdefault('sample_id',sample)
            scanner=next((part for part in path.relative_to(root).parts[:-1] if part.lower() in ('trios3','trios5','itero','labscanner','primescan')),'')
            info.setdefault('scanner',scanner);metadata[path]=info
        if manifest:
            with Path(manifest).open(encoding='utf-8-sig',newline='') as stream:
                for row in csv.DictReader(stream):
                    candidates=[(root/row['annotation_mesh']).resolve()] if row.get('annotation_mesh') else [p for p in queue if row.get('sample_dir') and p.parent==(root/row['sample_dir']).resolve()]
                    for path in candidates:
                        if path in metadata:metadata[path].update({k:v for k,v in row.items() if v is not None and v!=''})
        return metadata

    def sort_queue(self,*args):
        if not self.queue:return
        mode=self.sort_picker.currentData();current=self.mesh.path if self.mesh else None
        def natural(value):return tuple((0,int(part)) if part.isdigit() else (1,part) for part in re.split(r'(\d+)',str(value).casefold()))
        def key(path):
            info=self.queue_info.get(path,{});sample=natural(info.get('sample_id',path.parent.name));relative=natural(path.relative_to(self.root))
            if mode in ('sample','sample_desc'):return sample,relative
            if mode=='scanner':return natural(info.get('scanner','')),sample,relative
            if mode=='group':return natural(info.get('group','')),natural(info.get('physical_id','')),sample,relative
            if mode=='stage':return {'partial':0,'baseline':1,'clean':2}.get(info.get('stage'),3),sample,relative
            if mode=='status':return {'draft':0,'new':1,'error':2,'done':3}[self.store.status(path)],sample,relative
            return relative
        self.queue=list(self.queue_order) if mode=='original' else sorted(self.queue_order,key=key,reverse=mode=='sample_desc')
        self.index=self.queue.index(current) if current in self.queue else -1
        self.list.blockSignals(True);self.list.clear()
        for path in self.queue:self.list.addItem(str(path.relative_to(self.root)))
        if self.index>=0:self.list.setCurrentRow(self.index)
        self.list.blockSignals(False);self.refresh_queue()
        if self.mesh:self.sample_title.setText(f'{self.index+1} / {len(self.queue)}  •  {self.mesh.path.parent.name} / {self.mesh.path.name}')

    def set_scanner_color(self,enabled):
        self.view.color_enabled=enabled
        for control in self.color_controls.values():control.setEnabled(enabled)
        self.view.apply_color_adjustments()

    def install_feature_controls(self,main):
        colors=QtWidgets.QHBoxLayout();self.color_controls={};self.color_values={}
        for name,lo,hi,value in [('brightness',-50,50,0),('contrast',25,250,100),('gamma',40,250,100),('saturation',0,200,100)]:
            group=QtWidgets.QVBoxLayout();caption=QtWidgets.QHBoxLayout()
            caption.addWidget(QtWidgets.QLabel(name.title()));caption.addStretch()
            indicator=QtWidgets.QLabel(f'{value}%');indicator.setMinimumWidth(42)
            indicator.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignVCenter)
            self.color_values[name]=indicator;caption.addWidget(indicator);group.addLayout(caption)
            control=QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            control.setRange(lo,hi);control.setValue(value);control.setSingleStep(1);control.setPageStep(5)
            control.setMinimumWidth(90);control.setAccessibleName(name.title())
            control.setToolTip('Display only: original RGB/texture and exports are preserved.')
            control.valueChanged.connect(lambda v,n=name:self.color_values[n].setText(f'{v}%'))
            self.color_controls[name]=control;group.addWidget(control);colors.addLayout(group,1)
        button=QtWidgets.QPushButton('Reset colors');button.clicked.connect(self.reset_colors)
        colors.addWidget(button,0,QtCore.Qt.AlignmentFlag.AlignBottom)
        main.insertLayout(main.indexOf(self.view),colors)
        self.color_timer=QtCore.QTimer(self);self.color_timer.setSingleShot(True);self.color_timer.setInterval(120)
        self.color_timer.timeout.connect(self.update_display_colors)
        # Throttle rendering while dragging, so long drags still update the view.
        for control in self.color_controls.values():control.valueChanged.connect(self.schedule_color_update)
        row=QtWidgets.QHBoxLayout();self.top_guard=QtWidgets.QCheckBox('Paint from top only');self.top_guard.setChecked(True)
        self.top_guard.toggled.connect(self.set_top_guard);row.addWidget(self.top_guard)
        button=QtWidgets.QPushButton('Top view [R]');button.clicked.connect(self.view.reset_camera);row.addWidget(button)
        button=QtWidgets.QPushButton('Flip top/bottom');button.clicked.connect(self.flip_top);row.addWidget(button)
        self.orientation_note=QtWidgets.QLabel('TOP axis follows the sample');row.addWidget(self.orientation_note);row.addStretch()
        main.insertLayout(main.indexOf(self.view),row)

    def schedule_color_update(self,*args):
        if not self.color_timer.isActive():self.color_timer.start()

    def update_display_colors(self):
        for name,control in self.color_controls.items():self.color_values[name].setText(f'{control.value()}%')
        self.view.color_adjustments={name:control.value()/100 for name,control in self.color_controls.items()}
        self.view.apply_color_adjustments()

    def reset_colors(self):
        for name,control in self.color_controls.items():
            control.blockSignals(True);control.setValue(0 if name=='brightness' else 100);control.blockSignals(False)
        self.color_timer.stop();self.update_display_colors();self.view.setFocus()

    def set_top_guard(self,enabled):
        self.view.top_only=enabled;self.view.update_side_indicator();self.view.GetRenderWindow().Render()

    def flip_top(self):
        if self.mesh is None:return
        normal=-self.view.top_normal
        try:
            atomic_npz(orientation_path(self.store,self.mesh),fingerprint=np.array(self.mesh.fingerprint),top_normal=normal)
            self.view.set_top_normal(normal);self.orientation_note.setText('Shared top direction corrected')
            self.view.reset_camera();self.dirty=True;self.autosave()
        except Exception as exc:self.show_error(str(exc))

    def refresh_annotators(self,selected):
        names={selected}
        if self.store and self.store.output.is_dir():names.update(p.name for p in self.store.output.iterdir() if p.is_dir() and p.name!='_sample_orientation' and re.fullmatch(r'[A-Za-z0-9_-]{1,64}',p.name))
        self.rater.blockSignals(True);self.rater.clear();self.rater.addItems(sorted(names));self.rater.setCurrentText(selected);self.rater.blockSignals(False)

    def create_annotator(self,name=None):
        if name is None:
            name,ok=QtWidgets.QInputDialog.getText(self,'New annotator','Name / ID (letters, numbers, _ or -):')
            if not ok:return
        name=name.strip()
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',name) or name=='_sample_orientation':self.show_error('Use 1–64 letters, numbers, underscores or hyphens.');return
        if self.store:
            try:(self.store.output/name).mkdir(parents=True,exist_ok=True)
            except Exception as exc:self.show_error(str(exc));return
        if self.rater.findText(name)<0:self.rater.addItem(name)
        self.rater.setCurrentText(name)

    def switch_annotator(self,name):
        if not name:return
        if self.store is None:self.settings.setValue('annotator',name);return
        previous=self.store.annotator
        if name==previous:return
        if not self.save():self.restore_rater_choice(previous);return
        lock=None
        try:
            store=AnnotationStore(self.root,self.store.output,name)
            if self.mesh is not None:lock=self.acquire_lock(store.path_for(self.mesh.path));labels,completed,notes=store.load(self.mesh)
            self.release_lock();self.lock_path=lock;self.store=store
            if self.mesh is not None:
                self.labels,self.completed=labels,completed;self.undo_stack=[];self.redo_stack=[];self.stroke={};self.dirty=False
                self.notes.blockSignals(True);self.notes.setPlainText(notes);self.notes.blockSignals(False)
                self.last_save=f'Loaded {name} annotation';self.refresh()
            self.settings.setValue('annotator',name);self.refresh_queue();self.update_review_mode();self.view.setFocus()
        except Exception as exc:
            if lock and lock!=self.lock_path:lock.unlock()
            self.restore_rater_choice(previous);self.show_error(str(exc))

    def is_read_only(self):
        return bool(getattr(self,'review_owner',None) and self.store and self.store.annotator!=self.review_owner)

    def update_review_mode(self):
        readonly=self.is_read_only();self.notes.setReadOnly(readonly);self.mode.setEnabled(not readonly)
        self.clear_labels_button.setEnabled(not readonly)
        self.all_plaque_button.setEnabled(not readonly)
        self.roi_review_button.setEnabled(not readonly)
        self.mode.setChecked(not readonly)
        if readonly:self.notice.setText('READ ONLY: independent labels. Select '+self.review_owner+' to edit consensus.')
        elif self.mesh:self.notice.setText(self.mesh.warning or 'Scanner colors • adjustments affect the display only')

    def restore_rater_choice(self,name):
        self.rater.blockSignals(True);self.rater.setCurrentText(name);self.rater.blockSignals(False)

    def export_all(self):
        if self.store is None or not self.save():return
        folder=QtWidgets.QFileDialog.getExistingDirectory(self,'Export completed annotations in this queue for the current annotator',str(self.store.output))
        if not folder:return
        from export_annotations import export_annotations
        try:
            report=export_annotations(self.root,self.store.output,Path(folder),[self.store.annotator],self.queue)
            self.statusBar().showMessage(f'Exported {report["exported"]} PLY + label-color PLY + NPZ bundles; skipped {report["skipped"]}; failures {report["failed"]}')
        except Exception as exc:self.show_error(str(exc))
