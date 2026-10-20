"""Local BioCal3D surface annotator. Run: python biocal3d_annotator.py --demo"""
from __future__ import annotations
import argparse
import json
import sys
import traceback
from pathlib import Path
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from mesh_io import (LABELS, PALETTE, AnnotationStore, create_demo,
                     discover, manifest_queue, export_vtp, export_ply, export_label_colors, load_mesh, shaded_display)

from roi_review_store import read_reviewed, current_version, read_state, approve_displayed_surface, resolve_review_source, manifest_reviews
from surface_display import display_subset
from rotation_keys import SpaceRotateFilter

from annotation_features import SurfaceFeatures, AnnotatorFeatures, read_orientation

class SurfaceView(SurfaceFeatures, QVTKRenderWindowInteractor):
    stroke_started = QtCore.Signal()
    faces_painted = QtCore.Signal(object)
    stroke_finished = QtCore.Signal()
    brush_changed = QtCore.Signal(int)
    failure = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.075, 0.095, 0.13)
        self.GetRenderWindow().AddRenderer(self.renderer)
        self.GetRenderWindow().SetMultiSamples(0)
        self.GetRenderWindow().GetInteractor().SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.mapper = vtk.vtkPolyDataMapper()
        self.base = vtk.vtkActor(); self.base.SetMapper(self.mapper)
        self.base.GetProperty().LightingOff()
        self.base.GetProperty().SetColor(.8, .81, .82)
        self.renderer.AddActor(self.base)
        self.points_mapper=vtk.vtkPolyDataMapper()
        self.points_mapper.SetResolveCoincidentTopologyToPolygonOffset()
        self.points_mapper.SetRelativeCoincidentTopologyPointOffsetParameter(-2)
        self.points=vtk.vtkActor(); self.points.SetMapper(self.points_mapper)
        self.points.GetProperty().SetRepresentationToPoints(); self.points.GetProperty().SetPointSize(3)
        self.apply_point_style()
        self.points.PickableOff(); self.points.VisibilityOff()
        self.renderer.AddActor(self.points)
        self.wire_mapper=vtk.vtkPolyDataMapper(); self.wire_mapper.ScalarVisibilityOff()
        self.wire=vtk.vtkActor(); self.wire.SetMapper(self.wire_mapper)
        self.wire.GetProperty().SetRepresentationToWireframe(); self.wire.GetProperty().SetColor(.50,.61,.73)
        self.wire.GetProperty().LightingOff(); self.wire.PickableOff(); self.wire.VisibilityOff()
        self.wire_mapper.SetResolveCoincidentTopologyToPolygonOffset()
        self.wire_mapper.SetRelativeCoincidentTopologyLineOffsetParameters(-2,-2)
        self.renderer.AddActor(self.wire)
        self.overlay_mapper = vtk.vtkPolyDataMapper()
        self.overlay_mapper.SetScalarModeToUseCellData()
        self.overlay_mapper.SetColorModeToDirectScalars()
        self.overlay_mapper.SetResolveCoincidentTopologyToPolygonOffset()
        self.overlay_mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-1, -1)
        self.overlay = vtk.vtkActor(); self.overlay.SetMapper(self.overlay_mapper)
        self.overlay.GetProperty().LightingOff(); self.overlay.PickableOff()
        self.renderer.AddActor(self.overlay)
        self.circle_data = vtk.vtkPolyData()
        cm = vtk.vtkPolyDataMapper2D(); cm.SetInputData(self.circle_data)
        self.circle = vtk.vtkActor2D(); self.circle.SetMapper(cm)
        self.circle.GetProperty().SetColor(.98, .98, 1); self.circle.GetProperty().SetLineWidth(1.5)
        self.circle.PickableOff(); self.circle.VisibilityOff()
        self.renderer.AddActor(self.circle)
        self.mesh = None; self.radius = 24; self.paint_mode = True
        self.space_down = False; self.painting = False; self.previous = None
        self.overlay_enabled = True; self.overlay_alpha = .45
        self.surface_enabled=True; self.points_enabled=False; self.wire_enabled=False; self.shading='none'
        self.projection_key = None; self.projected = None
        self.setMouseTracking(True); self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.installEventFilter(self)
        self.init_features()
        self.rotation_keys=SpaceRotateFilter(self)
        self.Initialize()

    def set_mesh(self, mesh):
        self.mesh = mesh; self.projection_key = None
        self.display_face_ids=np.arange(len(mesh.faces),dtype=np.int64)
        self.setup_orientation(mesh)
        self.render_data=shaded_display(mesh)
        self.cache_colors()
        self.mapper.SetInputData(self.render_data)
        self.mapper.SetScalarModeToUsePointData(); self.mapper.SetColorModeToDirectScalars()
        self.mapper.SetScalarVisibility(mesh.display.GetPointData().GetScalars() is not None)
        self.points_mapper.SetInputData(self.render_data)
        self.apply_point_style()
        self.wire_mapper.SetInputData(self.render_data)
        self.overlay_data = vtk.vtkPolyData(); self.overlay_data.ShallowCopy(self.render_data)
        self.overlay_mapper.SetInputData(self.overlay_data)
        self.base.SetTexture(None)
        self.points.SetTexture(None)
        if mesh.texture and mesh.display.GetPointData().GetTCoords() is not None:
            self.set_texture(mesh.texture)
        else:
            self.apply_color_adjustments()
        self.centers = mesh.vertices[mesh.faces].mean(axis=1)
        self.locator = vtk.vtkStaticCellLocator(); self.locator.SetDataSet(self.render_data); self.locator.BuildLocator()
        self.picker = vtk.vtkCellPicker(); self.picker.SetTolerance(.0005)
        self.picker.PickFromListOn(); self.picker.AddPickList(self.base); self.picker.AddLocator(self.locator)
        self.apply_render_options()
        self.reset_camera()

    def set_display_faces(self,ids):
        data,mapping=display_subset(self.render_data,ids)
        self.display_face_ids=mapping
        for mapper in (self.mapper,self.points_mapper,self.wire_mapper):mapper.SetInputData(data)
        self.overlay_data=vtk.vtkPolyData();self.overlay_data.ShallowCopy(data)
        self.overlay_mapper.SetInputData(self.overlay_data)
        self.locator=vtk.vtkStaticCellLocator();self.locator.SetDataSet(data)
        if len(mapping):self.locator.BuildLocator()
        self.picker=vtk.vtkCellPicker();self.picker.SetTolerance(.0005)
        self.picker.PickFromListOn();self.picker.AddPickList(self.base)
        if len(mapping):self.picker.AddLocator(self.locator)
        self.projection_key=None

    def apply_render_options(self):
        # Reassert each actor's representation: VTK's default S/W bindings can
        # otherwise turn the points/wire actors into surfaces behind the UI.
        self.base.GetProperty().SetRepresentationToSurface()
        self.apply_point_style()
        self.wire.GetProperty().SetRepresentationToWireframe()
        self.base.SetVisibility(self.surface_enabled)
        self.points.SetVisibility(self.points_enabled)
        self.wire.SetVisibility(self.wire_enabled)
        prop=self.base.GetProperty()
        prop.SetLighting(self.shading!='none'); prop.SetAmbient(.30); prop.SetDiffuse(.70); prop.SetSpecular(0)
        if self.shading=='face':prop.SetInterpolationToFlat()
        else:prop.SetInterpolationToGouraud()
        # Show face labels in whichever mesh representation is currently visible.
        op=self.overlay.GetProperty()
        if self.surface_enabled:op.SetRepresentationToSurface()
        elif self.points_enabled:op.SetRepresentationToPoints(); op.SetPointSize(self.points.GetProperty().GetPointSize()+1)
        else:op.SetRepresentationToWireframe()
        self.overlay.SetVisibility(self.overlay_enabled and (self.surface_enabled or self.points_enabled or self.wire_enabled))
        self.GetRenderWindow().Render()

    def set_texture(self, path):
        self.load_display_texture(path)

    def reset_camera(self):
        self.reset_top_camera()

    def camera_state(self):
        c = self.renderer.GetActiveCamera()
        return {'position': c.GetPosition(), 'focal_point': c.GetFocalPoint(), 'view_up': c.GetViewUp(),
                'parallel_scale': c.GetParallelScale(), 'display_adjustments': self.color_adjustments.copy(),
                'top_normal': self.top_normal.tolist(), 'top_only': self.top_only,'scanner_color':self.color_enabled}

    def restore_camera(self, state):
        if not state:
            return
        c = self.renderer.GetActiveCamera()
        c.SetPosition(*state['position']); c.SetFocalPoint(*state['focal_point']); c.SetViewUp(*state['view_up'])
        c.SetParallelScale(state['parallel_scale']); self.renderer.ResetCameraClippingRange()
        self.GetRenderWindow().Render()

    def update_labels(self, labels):
        if self.mesh is None:
            return
        rgba = np.zeros((len(labels), 4), np.uint8)
        for value, color in PALETTE.items():
            mask = (labels == value) & self.mesh.active
            rgba[mask, :3] = color
            rgba[mask, 3] = 0 if value == -2 else int(self.overlay_alpha*255)
        rgba[~self.mesh.active] = [110, 115, 125, 150]
        a = numpy_to_vtk(rgba, deep=True); a.SetName('AnnotationOverlay')
        self.overlay_data.GetCellData().SetScalars(a)
        self.apply_render_options()
        self.GetRenderWindow().Render()

    def _position(self, event):
        w, h = self.GetRenderWindow().GetSize()
        sx, sy = w/max(self.width(),1), h/max(self.height(),1)
        return float(event.position().x()*sx), float((self.height()-event.position().y())*sy), sx

    def _circle(self, x, y, radius):
        t = np.linspace(0, 2*np.pi, 65)
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(np.column_stack((x+radius*np.cos(t), y+radius*np.sin(t), np.zeros(len(t)))), deep=True))
        lines = vtk.vtkCellArray(); lines.InsertNextCell(len(t))
        for i in range(len(t)):
            lines.InsertCellPoint(i)
        self.circle_data.SetPoints(points); self.circle_data.SetLines(lines)
        self.circle.SetVisibility(self.paint_mode and not self.space_down)

    def _project_centers(self):
        w, h = self.GetRenderWindow().GetSize()
        c = self.renderer.GetActiveCamera()
        key = (c.GetMTime(), w, h)
        if key != self.projection_key:
            m = c.GetCompositeProjectionTransformMatrix(self.renderer.GetTiledAspectRatio(), 0, 1)
            mat = np.array([[m.GetElement(i,j) for j in range(4)] for i in range(4)])
            pos = np.column_stack((self.centers, np.ones(len(self.centers)))) @ mat.T
            with np.errstate(divide='ignore', invalid='ignore'):
                ndc = pos[:,:3]/pos[:,3,None]
            self.projected = np.column_stack(((ndc[:,0]+1)*w/2, (ndc[:,1]+1)*h/2))
            self.projection_key = key
        return self.projected

    def brush_faces(self, x, y, radius):
        """Visible triangles only; circular membership uses projected triangle centroids."""
        if self.mesh is None:
            return np.empty(0, np.int64)
        if getattr(self,'top_only',False) and self.side_alignment()<=.08:
            return np.empty(0,np.int64)
        mapping=getattr(self,'display_face_ids',None)
        if mapping is not None and not len(mapping):return np.empty(0,np.int64)
        w, h = self.GetRenderWindow().GetSize()
        if w < 2 or h < 2:
            return np.empty(0, np.int64)
        selection = vtk.vtkHardwareSelector(); selection.SetRenderer(self.renderer)
        selection.SetFieldAssociation(vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS)
        selection.SetArea(max(0, int(x-radius)), max(0, int(y-radius)),
                          min(w-1, int(x+radius)), min(h-1, int(y+radius)))
        # Use a solid surface for visibility/picking in every display mode. Do not
        # change the IDs or allow sparse point/wire rendering to pick hidden faces.
        actors=[self.base,self.overlay]+[getattr(self,n) for n in ('points','wire','side_text') if hasattr(self,n)]+list(getattr(self,'review_guides',[]))
        old_visibility=[a.GetVisibility() for a in actors]
        self.overlay.VisibilityOff(); self.circle.VisibilityOff(); self.base.VisibilityOn()
        for actor in actors[2:]:actor.VisibilityOff()
        try:
            result = selection.Select()
            ids = []
            if result:
                for i in range(result.GetNumberOfNodes()):
                    node = result.GetNode(i)
                    if node.GetProperties().Get(vtk.vtkSelectionNode.PROP()) == self.base:
                        ids.extend(vtk_to_numpy(node.GetSelectionList()).tolist())
            ids = np.unique(ids).astype(np.int64)
            count=len(mapping) if mapping is not None else len(self.mesh.faces)
            ids = ids[(ids >= 0) & (ids < count)]
            if mapping is not None:ids=mapping[ids]
            projected = self._project_centers()
            if len(ids):
                ids = ids[((projected[ids]-[x,y])**2).sum(axis=1) <= radius**2]
            # Always include the surface directly under the cursor (also works on large triangles).
            self.picker.Pick(x, y, 0, self.renderer)
            hit = self.picker.GetCellId()
            if 0 <= hit < count:
                if mapping is not None:hit=int(mapping[hit])
                ids = np.unique(np.append(ids, hit))
            return ids[self.mesh.active[ids]]
        finally:
            for actor,visibility in zip(actors,old_visibility):actor.SetVisibility(visibility)

    def eventFilter(self, obj, event):
        kind = event.type()
        if kind in (QtCore.QEvent.Type.KeyPress, QtCore.QEvent.Type.KeyRelease) and event.key() == QtCore.Qt.Key.Key_Space:
            if not event.isAutoRepeat():self.set_space_down(kind == QtCore.QEvent.Type.KeyPress)
            return True
        # Qt window shortcuts handle our keys before KeyPress is delivered.
        # Suppress VTK's additional CharEvent shortcuts (S/W/R/F/Q/etc.) so
        # keyboard input cannot silently change actor modes, cameras or exit.
        if kind in (QtCore.QEvent.Type.KeyPress,QtCore.QEvent.Type.KeyRelease):return True
        if kind == QtCore.QEvent.Type.FocusOut:
            self.set_space_down(False)
            if self.painting:
                self.painting = False; self.stroke_finished.emit()
        if kind == QtCore.QEvent.Type.Leave:
            self.circle.VisibilityOff(); self.GetRenderWindow().Render()
        if kind == QtCore.QEvent.Type.MouseButtonPress and event.button() == QtCore.Qt.MouseButton.LeftButton:
            if self.paint_mode and not self.space_down and self.mesh is not None:
                if not (self.surface_enabled or self.points_enabled or self.wire_enabled):
                    return True
                self.setFocus(); self.painting = True; self.previous = None
                self.stroke_started.emit(); self._paint_event(event); return True
        if kind == QtCore.QEvent.Type.MouseMove and self.mesh is not None:
            x, y, scale = self._position(event)
            self._circle(x,y,self.radius*scale)
            if self.painting:
                self._paint_event(event); return True
            if not event.buttons():
                self.GetRenderWindow().Render()
        if kind == QtCore.QEvent.Type.MouseButtonRelease and event.button() == QtCore.Qt.MouseButton.LeftButton and self.painting:
            self.painting = False; self.previous = None; self.stroke_finished.emit()
            return True
        if kind == QtCore.QEvent.Type.Wheel and event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            self.radius = int(np.clip(self.radius+(3 if event.angleDelta().y()>0 else -3), 3, 150))
            self.brush_changed.emit(self.radius); return True
        return super().eventFilter(obj,event)

    def set_space_down(self,enabled):
        if self.space_down==enabled:return
        self.space_down=enabled
        if enabled and self.painting:
            self.painting=False;self.previous=None;self.stroke_finished.emit()
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor if enabled else
                       (QtCore.Qt.CursorShape.CrossCursor if self.paint_mode else QtCore.Qt.CursorShape.ArrowCursor))
        if enabled:self.circle.VisibilityOff()
        self.GetRenderWindow().Render()

    def _paint_event(self, event):
        x,y,scale = self._position(event); radius = self.radius*scale
        prev = self.previous if self.previous is not None else (x,y)
        count = max(1, int(np.ceil(np.linalg.norm(np.array(prev)-[x,y])/max(radius*.5,1))))
        try:
            ids = [self.brush_faces(px,py,radius) for px,py in np.linspace(prev,(x,y),count+1)[1:]]
            if ids:
                self.faces_painted.emit(np.unique(np.concatenate(ids)))
        except Exception as exc:
            self.painting = False; self.stroke_finished.emit(); self.failure.emit(str(exc))
        self.previous = (x,y); self._circle(x,y,radius); self.GetRenderWindow().Render()

class Annotator(AnnotatorFeatures, QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle('BioCal3D • 3D surface labeling'); self.resize(1380,880)
        self.review_owner=args.annotator if getattr(args,'review_mode',False) else None
        self.root = None; self.store = None; self.queue = []; self.queue_order=[];self.queue_info={};self.index = -1
        self.mesh = None; self.labels = None; self.completed = False; self.dirty = False
        self.undo_stack = []; self.redo_stack = []; self.stroke = {}; self.lock_path = None
        self.shortcuts = []; self.last_save = 'Not saved'
        self.settings = QtCore.QSettings('BioCal3D','SurfaceAnnotator')
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central); layout.setContentsMargins(14,14,14,14)
        sidebar = QtWidgets.QWidget(); sidebar.setFixedWidth(300); side = QtWidgets.QVBoxLayout(sidebar)
        title = QtWidgets.QLabel('BioCal3D'); title.setStyleSheet('font-size:26px;font-weight:700;color:#edf4ff')
        side.addWidget(title); side.addWidget(QtWidgets.QLabel('SURFACE ANNOTATION'))
        rater_row=QtWidgets.QHBoxLayout()
        self.rater=QtWidgets.QComboBox(); self.rater.addItem(args.annotator or self.settings.value('annotator','rater1'))
        self.rater.currentTextChanged.connect(self.switch_annotator); rater_row.addWidget(self.rater,1)
        button=QtWidgets.QPushButton('New…'); button.clicked.connect(lambda:self.create_annotator()); rater_row.addWidget(button)
        side.addLayout(rater_row)
        if getattr(args,'lock_annotator',False):self.rater.setEnabled(False);button.setEnabled(False)
        self.folder_label = QtWidgets.QLabel('Choose your prepared ROI folder'); self.folder_label.setWordWrap(True)
        side.addWidget(self.folder_label)
        self.filter = QtWidgets.QLineEdit(); self.filter.setPlaceholderText('Optional path filter, e.g. G3 or T1')
        side.addWidget(self.filter)
        self.open_button = QtWidgets.QPushButton('Open dataset folder…'); self.open_button.clicked.connect(self.open_folder)
        side.addWidget(self.open_button)
        sort_row=QtWidgets.QHBoxLayout();sort_row.addWidget(QtWidgets.QLabel('Sort by'))
        self.sort_picker=QtWidgets.QComboBox()
        for text,key in [('Dataset order','original'),('Sample A–Z','sample'),('Sample Z–A','sample_desc'),('Scanner','scanner'),('Group / specimen','group'),('Stage: partial first','stage'),('Label status','status')]:
            self.sort_picker.addItem(text,key)
        self.sort_picker.currentIndexChanged.connect(self.sort_queue);sort_row.addWidget(self.sort_picker,1)
        resort=QtWidgets.QPushButton('↻');resort.setToolTip('Reapply sorting, including updated annotation statuses')
        resort.setFixedWidth(34);resort.clicked.connect(self.sort_queue);sort_row.addWidget(resort);side.addLayout(sort_row)
        self.list = QtWidgets.QListWidget(); self.list.currentRowChanged.connect(self.change_sample); side.addWidget(self.list,1)
        self.progress = QtWidgets.QLabel('0 samples');self.progress.setWordWrap(True); side.addWidget(self.progress)
        self.notes = QtWidgets.QPlainTextEdit(); self.notes.setPlaceholderText('Optional annotation notes')
        self.notes.setMaximumHeight(90); self.notes.textChanged.connect(self.note_changed); side.addWidget(self.notes)
        layout.addWidget(sidebar)
        main = QtWidgets.QVBoxLayout(); layout.addLayout(main,1)
        self.sample_title = QtWidgets.QLabel('Load a dataset to begin'); self.sample_title.setStyleSheet('font-size:19px;font-weight:600')
        self.sample_title.setWordWrap(True)
        main.addWidget(self.sample_title)
        self.notice = QtWidgets.QLabel('Labels are saved separately. Original scanner colors and geometry are preserved.')
        self.notice.setWordWrap(True); self.notice.setStyleSheet('color:#a9b9cd;padding-bottom:5px'); main.addWidget(self.notice)
        toolbar = QtWidgets.QHBoxLayout()
        self.class_picker = QtWidgets.QComboBox()
        for value in (1,0,-1,-2):
            color = QtGui.QPixmap(14,14); color.fill(QtGui.QColor(*PALETTE[value]))
            self.class_picker.addItem(QtGui.QIcon(color), f'{LABELS[value].title()}  [{ {1:"1",0:"2",-1:"3",-2:"4"}[value]}]', value)
        toolbar.addWidget(self.class_picker)
        self.mode = QtWidgets.QCheckBox('Paint mode'); self.mode.setChecked(True)
        self.mode.toggled.connect(self.set_mode); toolbar.addWidget(self.mode)
        toolbar.addWidget(QtWidgets.QLabel('Brush'))
        self.radius = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal); self.radius.setRange(3,150); self.radius.setValue(24)
        self.radius.setMaximumWidth(150); toolbar.addWidget(self.radius)
        self.radius_text = QtWidgets.QLabel('24 px'); toolbar.addWidget(self.radius_text)
        self.show_overlay = QtWidgets.QCheckBox('Labels [O]'); self.show_overlay.setChecked(True)
        self.show_overlay.toggled.connect(self.refresh); toolbar.addWidget(self.show_overlay)
        self.opacity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal); self.opacity.setRange(10,90); self.opacity.setValue(45)
        self.opacity.setToolTip('Label overlay opacity'); self.opacity.setMaximumWidth(90)
        self.opacity.valueChanged.connect(self.refresh); toolbar.addWidget(self.opacity)
        toolbar.addStretch(); main.addLayout(toolbar)
        rendering=QtWidgets.QHBoxLayout()
        self.surface_toggle=QtWidgets.QCheckBox('Surface [S]'); self.surface_toggle.setChecked(True)
        self.points_toggle=QtWidgets.QCheckBox('Points [P]')
        self.wire_toggle=QtWidgets.QCheckBox('Wireframe [W]')
        self.points_toggle.setToolTip('Show yellow mesh points. Turn Surface off for a points-only view.')
        self.wire_toggle.setToolTip('Show triangle edges. Turn Surface off for a wireframe-only view.')
        for control in (self.surface_toggle,self.points_toggle,self.wire_toggle):rendering.addWidget(control)
        self.color_toggle=QtWidgets.QCheckBox('Scanner color');self.color_toggle.setChecked(True)
        self.color_toggle.setToolTip('Show original RGB/texture, or neutral gray geometry. Annotation labels and exports are preserved.')
        rendering.addWidget(self.color_toggle)
        rendering.addWidget(QtWidgets.QLabel('Shading'))
        self.shading_picker=QtWidgets.QComboBox()
        for label,value in [('None (original color)','none'),('Vertex (smooth)','vertex'),('Face (flat)','face')]:
            self.shading_picker.addItem(label,value)
        rendering.addWidget(self.shading_picker)
        rendering.addWidget(QtWidgets.QLabel('Point size'))
        self.point_size=QtWidgets.QSpinBox(); self.point_size.setRange(1,12); self.point_size.setValue(3)
        rendering.addWidget(self.point_size); rendering.addStretch()
        self.auto_ply=QtWidgets.QCheckBox('Auto-save PLY too');self.auto_ply.setChecked(True)
        self.auto_ply.setToolTip('Also save the original-color training PLY. A separate label-colored viewing PLY is always saved with the NPZ.')
        rendering.addWidget(self.auto_ply);main.addLayout(rendering)
        self.view = SurfaceView(); main.addWidget(self.view,1)
        self.install_feature_controls(main)
        review_row=QtWidgets.QHBoxLayout();review_row.setSpacing(6)
        self.roi_review_button=QtWidgets.QPushButton('Verify / correct ROI…')
        self.roi_review_button.setToolTip('Open the original-model context and extraction/manual correction window.')
        self.roi_review_button.clicked.connect(self.open_roi_review)
        self.approve_roi_button=QtWidgets.QPushButton('Approve ROI && Next [Ctrl+Enter]')
        self.approve_roi_button.setObjectName('primary');self.approve_roi_button.setEnabled(False)
        self.approve_roi_button.setToolTip('Ctrl+Enter: approve the displayed ROI as-is and open the next ROI needing review. Plaque labels are preserved.')
        self.approve_roi_button.clicked.connect(self.approve_roi_next)
        self.next_roi_button=QtWidgets.QPushButton('Next unapproved')
        self.next_roi_button.setToolTip('Move to the next ROI needing review without approving this sample.')
        self.next_roi_button.clicked.connect(self.next_roi_review)
        for button in (self.roi_review_button,self.next_roi_button,self.approve_roi_button):
            button.setProperty('roiControl',True);button.setFixedHeight(32)
            review_row.addWidget(button)
        review_row.addStretch(1)
        self.roi_status=QtWidgets.QLabel('ROI: pending');self.roi_status.setObjectName('roiStatus')
        self.roi_status.setToolTip('Surface review: pending • original crop. F5 reloads the latest shared ROI.')
        review_row.addWidget(self.roi_status)
        self.label_status=QtWidgets.QLabel('Label: pending');self.label_status.setObjectName('labelStatus')
        self.label_status.setToolTip('Plaque-label approval for the selected annotator; independent of ROI approval.')
        review_row.addWidget(self.label_status)
        main.insertLayout(main.indexOf(self.view),review_row)
        self.color_toggle.toggled.connect(self.set_scanner_color)
        for control in (self.surface_toggle,self.points_toggle,self.wire_toggle):control.toggled.connect(self.render_options_changed)
        self.shading_picker.currentIndexChanged.connect(self.render_options_changed)
        self.point_size.valueChanged.connect(self.render_options_changed)
        self.class_picker.activated.connect(lambda *_: self.view.setFocus())
        self.radius.valueChanged.connect(self.set_radius); self.view.brush_changed.connect(self.radius.setValue)
        self.view.stroke_started.connect(self.start_stroke); self.view.faces_painted.connect(self.paint_faces)
        self.view.stroke_finished.connect(self.end_stroke); self.view.failure.connect(self.show_error)
        tools = QtWidgets.QHBoxLayout()
        for text, callback in [('Undo [Ctrl+Z]',self.undo),('Redo [Ctrl+Y]',self.redo),('Reset view [R]',self.view.reset_camera)]:
            b = QtWidgets.QPushButton(text); b.clicked.connect(callback); tools.addWidget(b)
            if callback==self.redo:b.setToolTip('Redo annotation: Ctrl+Y or Ctrl+Shift+Z')
        tools.addStretch(1)
        bulk_actions=tools
        b = QtWidgets.QPushButton('Remaining → clean'); b.setToolTip('Mark only unreviewed ROI faces clean after reviewing the full surface.')
        b.clicked.connect(self.fill_clean);bulk_actions.addWidget(b)
        self.all_plaque_button=QtWidgets.QPushButton('Entire ROI → plaque')
        self.all_plaque_button.setToolTip('Replace every included ROI face label with Plaque. Excluded geometry is unchanged. Undo with Ctrl+Z.')
        self.all_plaque_button.clicked.connect(self.fill_plaque);bulk_actions.addWidget(self.all_plaque_button)
        main.addLayout(tools)
        actions = QtWidgets.QHBoxLayout()
        self.clear_labels_button = QtWidgets.QPushButton('Clear all labels')
        self.clear_labels_button.setToolTip('Reset all ROI faces to Unreviewed and mark this sample Draft. Undo with Ctrl+Z.')
        self.clear_labels_button.clicked.connect(self.clear_labels); actions.addWidget(self.clear_labels_button)
        b = QtWidgets.QPushButton('Save draft [Ctrl+S]'); b.clicked.connect(self.save); actions.addWidget(b)
        b = QtWidgets.QPushButton('Next unfinished'); b.clicked.connect(self.next_unfinished); actions.addWidget(b)
        b = QtWidgets.QPushButton('Approve Label && Next [N]'); b.setObjectName('primary'); b.clicked.connect(self.complete_next); actions.addWidget(b)
        main.addLayout(actions)
        help_text = QtWidgets.QLabel('Paint: left drag  •  Hold Space + left drag: rotate  •  Middle drag: pan  •  Wheel: zoom  •  Ctrl+wheel: brush size  •  A/D: previous/next sample')
        help_text.setWordWrap(True); help_text.setStyleSheet('color:#9daec4'); main.addWidget(help_text)
        for key, callback in [('1',lambda:self.class_picker.setCurrentIndex(0)),('2',lambda:self.class_picker.setCurrentIndex(1)),
                              ('3',lambda:self.class_picker.setCurrentIndex(2)),('4',lambda:self.class_picker.setCurrentIndex(3)),
                              ('Ctrl+Z',self.undo),('Ctrl+Y',self.redo),('Ctrl+Shift+Z',self.redo),('Ctrl+S',self.save),('N',self.complete_next),
                              ('Ctrl+Return',self.approve_roi_next),('Ctrl+Enter',self.approve_roi_next),('F5',self.reload_shared_roi),
                              ('R',self.view.reset_camera),('O',self.show_overlay.toggle),('B',self.mode.toggle),
                              ('S',self.surface_toggle.toggle),('W',self.wire_toggle.toggle),('P',self.points_toggle.toggle),
                              ('A',lambda:self.navigate_sample(-1)),('D',lambda:self.navigate_sample(1)),
                              ('[',lambda:self.radius.setValue(self.radius.value()-3)),(']',lambda:self.radius.setValue(self.radius.value()+3))]:
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            if key in ('1','2','3','4'):
                shortcut.setKeys([QtGui.QKeySequence(key),QtGui.QKeySequence('Num+'+key)])
            shortcut.setContext(QtCore.Qt.ShortcutContext.WindowShortcut)
            if key in ('Ctrl+Return','Ctrl+Enter'):shortcut.setAutoRepeat(False)
            shortcut.activated.connect(callback); self.shortcuts.append(shortcut)
        QtWidgets.QApplication.instance().focusChanged.connect(self.update_shortcuts)
        # Capture the held rotation key even when sliders/buttons have focus.
        QtWidgets.QApplication.instance().installEventFilter(self)
        self.timer = QtCore.QTimer(self); self.timer.setInterval(10000); self.timer.timeout.connect(self.autosave); self.timer.start()
        self.setStyleSheet('''QWidget {background:#111923;color:#dfe8f5;font-size:13px;}
            QLineEdit,QPlainTextEdit,QListWidget,QComboBox {background:#192534;border:1px solid #344359;border-radius:5px;padding:6px;}
            QPushButton {background:#24354a;border:1px solid #3c516b;border-radius:5px;padding:9px;}
            QPushButton:hover {background:#314a65;} QPushButton#primary {background:#16675b;border-color:#269d89;}
            QPushButton#primary:hover {background:#238574;border-color:#54c7ae;}
            QPushButton[roiControl="true"] {padding:4px 10px;}
            QLabel#roiStatus,QLabel#labelStatus {color:#c4cfde;padding:4px 8px;border:1px solid #344359;border-radius:5px;}
            QListWidget::item {padding:8px;} QListWidget::item:selected {background:#304c6b;}
            QSlider::groove:horizontal {height:5px;background:#344359;} QSlider::handle:horizontal {width:12px;background:#6ed6bc;margin:-4px 0;border-radius:6px;}''')
        self.statusBar().showMessage('Ready'); self.view.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        if args.demo:
            demo = create_demo(Path(__file__).parent/'demo_samples')
            QtCore.QTimer.singleShot(100,lambda:self.load_folder(demo,args.output))
        elif args.root:
            QtCore.QTimer.singleShot(100,lambda:self.load_folder(Path(args.root),args.output,getattr(args,'manifest',None)))

    def show_error(self, text):
        self.statusBar().showMessage('Error: '+text)
        QtWidgets.QMessageBox.warning(self,'BioCal3D',text)

    def eventFilter(self,obj,event):
        view=getattr(self,'view',None)
        if view is None:return super().eventFilter(obj,event)
        kind=event.type()
        if kind==QtCore.QEvent.Type.ApplicationDeactivate or (kind==QtCore.QEvent.Type.WindowDeactivate and obj is self):
            view.set_space_down(False)
        if isinstance(obj,QtWidgets.QWidget) and obj.window() is self:
            if kind in (QtCore.QEvent.Type.KeyPress,QtCore.QEvent.Type.KeyRelease) and event.key()==QtCore.Qt.Key.Key_Space:
                if kind==QtCore.QEvent.Type.KeyRelease and view.space_down:
                    if not event.isAutoRepeat():view.set_space_down(False)
                    return True
                current=QtWidgets.QApplication.focusWidget()
                editing=isinstance(current,(QtWidgets.QLineEdit,QtWidgets.QPlainTextEdit,QtWidgets.QTextEdit,QtWidgets.QAbstractSpinBox))
                modified=event.modifiers() & (QtCore.Qt.KeyboardModifier.ControlModifier|QtCore.Qt.KeyboardModifier.AltModifier|QtCore.Qt.KeyboardModifier.MetaModifier)
                if kind==QtCore.QEvent.Type.KeyPress and not editing and not modified:
                    view.setFocus();view.set_space_down(True)
                    return True
        return super().eventFilter(obj,event)

    def update_shortcuts(self,old=None,current=None):
        current=QtWidgets.QApplication.focusWidget()
        editing=isinstance(current,(QtWidgets.QLineEdit,QtWidgets.QPlainTextEdit,QtWidgets.QTextEdit,QtWidgets.QAbstractSpinBox))
        for shortcut in self.shortcuts:
            shortcut.setEnabled(not editing)

    def render_options_changed(self,*args):
        self.view.surface_enabled=self.surface_toggle.isChecked()
        self.view.points_enabled=self.points_toggle.isChecked()
        self.view.wire_enabled=self.wire_toggle.isChecked()
        self.view.shading=self.shading_picker.currentData()
        self.view.points.GetProperty().SetPointSize(self.point_size.value())
        self.view.apply_render_options()

    def open_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self,'Select prepared ROI dataset',self.settings.value('root',''))
        if folder:
            self.load_folder(Path(folder))

    def load_folder(self, root, output=None, manifest=None):
        if not self.save():
            return
        try:
            root = root.resolve()
            store = AnnotationStore(root, output or root/'annotations', self.rater.currentText().strip())
            queue = manifest_queue(root,manifest,self.filter.text().strip()) if manifest else discover(root,self.filter.text().strip())
            queue = list(dict.fromkeys(resolve_review_source(store,p) for p in queue if not p.is_relative_to(store.output)))
            if not queue:
                raise ValueError('No supported meshes found. Select the prepared ROI folder; accepted files are roi.vtp, geometry.npz, or triangle PLY/VTP.')
            self.release_lock()
            info=self.queue_metadata(root,queue,manifest)
            self.root,self.store,self.queue = root,store,queue
            self.queue_order=list(queue);self.queue_info=info
            self.mesh=None; self.labels=None; self.index=-1; self.dirty=False
            self.sort_queue()
            self.refresh_annotators(store.annotator); self.filter.setEnabled(False)
            self.folder_label.setText(str(root)); self.settings.setValue('root',str(root)); self.settings.setValue('annotator',store.annotator)
            self.refresh_queue()
            first = next((i for i,p in enumerate(self.queue) if store.status(p)!='done'),0)
            self.list.setCurrentRow(first)
        except Exception as exc:
            self.show_error(str(exc))

    def acquire_lock(self, path):
        lock = QtCore.QLockFile(str(path)+'.lock'); lock.setStaleLockTime(30000)
        path.parent.mkdir(parents=True,exist_ok=True)
        if not lock.tryLock(0):
            raise ValueError('This annotation is open in another app instance. Close that instance or choose a different annotator ID.')
        return lock

    def release_lock(self):
        if self.lock_path:
            self.lock_path.unlock(); self.lock_path=None

    def change_sample(self, index):
        if index<0 or index==self.index:
            return
        previous=self.index
        if not self.save():
            self.list.blockSignals(True); self.list.setCurrentRow(previous); self.list.blockSignals(False); return
        lock=None
        try:
            mesh=read_reviewed(self.store,self.queue[index]); lock=self.acquire_lock(self.store.path_for(mesh.path))
            labels,completed,notes=self.store.load(mesh)
            saved=self.store.path_for(mesh.path)
            camera={}
            if saved.exists():
                with np.load(saved,allow_pickle=False) as d:
                    camera=json.loads(str(d['camera'].item()))
                    texture=str(d['texture_path'].item()) if 'texture_path' in d else ''
                    if texture and Path(texture).is_file() and mesh.display.GetPointData().GetTCoords() is not None:
                        mesh.texture=Path(texture); mesh.warning=''
            self.view.set_mesh(mesh)
            normal=read_orientation(self.store,mesh)
            if normal is not None:
                self.view.set_top_normal(normal); self.view.top_source='saved shared top direction'; self.view.reset_camera()
            self.view.restore_camera(camera)
            self.release_lock(); self.lock_path=lock
            self.mesh,self.labels,self.completed,self.index=mesh,labels,completed,index
            self.undo_stack=[]; self.redo_stack=[]; self.stroke={}; self.dirty=False
            self.notes.blockSignals(True); self.notes.setPlainText(notes); self.notes.blockSignals(False)
            self.sample_title.setText(f'{index+1} / {len(self.queue)}  •  {mesh.path.parent.name} / {mesh.path.name}')
            self.notice.setText(mesh.warning or 'Scanner colors • adjustments affect the display only')
            self.last_save='Reopened annotation' if saved.exists() else 'New sample — not saved'
            self.update_roi_status(mesh)
            self.refresh(); self.update_review_mode(); self.view.setFocus()
        except Exception as exc:
            if lock and lock != self.lock_path:
                lock.unlock()
            if self.mesh is not None and self.view.mesh is not self.mesh:
                self.view.set_mesh(self.mesh); self.refresh()
            self.show_error(f'{self.queue[index]}\n\n{exc}')
            self.list.blockSignals(True); self.list.setCurrentRow(previous); self.list.blockSignals(False)

    def open_roi_review(self):
        if self.mesh is None or self.is_read_only() or not self.save():return
        from roi_review_ui import ReviewDialog
        self.timer.stop();dialog=None
        try:
            dialog=ReviewDialog(self)
            if dialog.exec() and dialog.saved:self.reload_shared_roi()
        except Exception as e:self.show_error(str(e))
        finally:
            if dialog is not None:dialog.deleteLater()
            self.timer.start()

    def update_roi_status(self,mesh):
        self.roi_status.setText('ROI: '+mesh.review_status)
        self.roi_status.setToolTip('Surface review: '+mesh.review_status+' • '+(mesh.review_version or 'original crop')+'. F5 reloads the latest shared ROI.')

    def approve_roi_next(self):
        if self.mesh is None or self.is_read_only() or not self.save():return
        self.timer.stop()
        try:
            record=approve_displayed_surface(self.store,self.mesh)
            # Approval preserves exact layout. Keep labels, completion, history,
            # notes and camera; adopt only the new shared review identity.
            self.mesh.review_version=record['version'];self.mesh.review_status='approved'
            self.update_roi_status(self.mesh)
            self.dirty=True
            if not self.save():return
            self.next_roi_review()
        except Exception as e:self.show_error('ROI approval failed. The current sample is kept open.\n\n'+str(e))
        finally:self.timer.start()

    def next_roi_review(self):
        if not self.queue:return
        # Geometry review has its own queue, independent of plaque completion.
        order=list(range(self.index+1,len(self.queue)))+list(range(0,max(self.index,0)))
        target=next((i for i in order if (read_state(self.store,self.queue[i]) or {}).get('status')!='approved'),None)
        if target is not None:self.list.setCurrentRow(target)
        else:
            self.refresh_queue()
            current_pending=self.mesh is not None and (read_state(self.store,self.mesh.path) or {}).get('status')!='approved'
            message=('No other ROI needs review; the current ROI still needs approval.' if current_pending else
                     'All ROIs in this queue are approved. Run rebuild_reviewed.cmd to export the verified dataset.')
            if not current_pending and (self.root/'scan_manifest.csv').is_file():
                try:
                    scans=manifest_reviews(self.store);remaining=sum(r['status']!='approved' for r in scans)
                    message=(f'Queue approved, but {remaining}/{len(scans)} exported ROIs still need review. Open the complete snapshot without a manifest/filter.' if remaining else
                             f'All {len(scans)} exported ROIs are approved. Run rebuild_reviewed.cmd. Reviews: {self.store.output}')
                except Exception as e:
                    self.show_error('Could not verify the complete dataset review status.\n\n'+str(e));return
            self.statusBar().showMessage(message)

    def reload_shared_roi(self):
        if self.mesh is None:return
        from roi_review_store import source_face_ids,source_domain,migrate_labels
        try:
            old=self.mesh;mesh=read_reviewed(self.store,old.path)
            if self.is_read_only():labels,completed,notes=self.store.load(mesh)
            elif mesh.fingerprint==old.fingerprint:
                labels=self.labels.copy();completed=False;notes=self.notes.toPlainText()
            else:
                ids=source_face_ids(old)
                data=dict(face_labels=self.labels,fingerprint=np.array(old.fingerprint),
                          source_face_ids=ids if ids is not None else np.arange(len(old.faces)),
                          source_domain=np.array(source_domain(old)),notes=np.array(self.notes.toPlainText()))
                labels,completed,notes=migrate_labels(self.store,mesh,data)
            self.mesh,self.labels,self.completed=mesh,labels,completed
            self.undo_stack=[];self.redo_stack=[];self.stroke={}
            self.notes.blockSignals(True);self.notes.setPlainText(notes);self.notes.blockSignals(False)
            self.view.set_mesh(mesh)
            normal=read_orientation(self.store,mesh)
            if normal is not None:self.view.set_top_normal(normal);self.view.reset_camera()
            self.update_roi_status(mesh)
            self.dirty=not self.is_read_only();self.refresh()
            if self.dirty:self.save()
        except Exception as e:self.show_error('Could not reload shared ROI. Current labels are kept open.\n\n'+str(e))

    def set_mode(self, enabled):
        self.view.paint_mode=enabled
        self.view.setCursor(QtCore.Qt.CursorShape.CrossCursor if enabled else QtCore.Qt.CursorShape.ArrowCursor)
        if not enabled:
            self.view.circle.VisibilityOff(); self.view.GetRenderWindow().Render()

    def set_radius(self,value):
        self.view.radius=value; self.radius_text.setText(f'{value} px')

    def start_stroke(self):
        self.stroke={}

    def paint_faces(self,ids):
        if self.is_read_only():return
        if self.labels is None or not len(ids):
            return
        value=self.class_picker.currentData()
        ids=np.asarray(ids,dtype=np.int64); ids=ids[self.labels[ids]!=value]
        for i in ids:
            self.stroke.setdefault(int(i),int(self.labels[i]))
        if len(ids):
            self.labels[ids]=value; self.completed=False; self.dirty=True; self.refresh()

    def end_stroke(self):
        if self.stroke:
            ids=np.array(list(self.stroke),dtype=np.int64)
            before=np.array(list(self.stroke.values()),dtype=np.int8)
            after=self.labels[ids].copy(); changed=before!=after
            if changed.any():
                self.undo_stack.append((ids[changed],before[changed],after[changed]))
                self.undo_stack=self.undo_stack[-100:]; self.redo_stack=[]
            self.stroke={}
            QtCore.QTimer.singleShot(1000,self.autosave)

    def edit(self,ids,value):
        if self.is_read_only():return
        if self.labels is None:
            return
        ids=np.asarray(ids); ids=ids[self.labels[ids]!=value]
        if not len(ids):
            return
        before=self.labels[ids].copy(); self.labels[ids]=value
        self.undo_stack.append((ids,before,self.labels[ids].copy())); self.redo_stack=[]
        self.completed=False; self.dirty=True; self.refresh(); self.autosave()

    def clear_labels(self):
        if self.is_read_only() or self.mesh is None or self.labels is None:return
        self.view.painting=False; self.view.previous=None
        self.end_stroke()
        self.edit(np.flatnonzero(self.mesh.active), -2)
        self.view.setFocus()

    def fill_clean(self):
        if self.is_read_only():return
        if self.view.top_only and self.mesh is not None and self.view.side_alignment()<=.08:
            self.show_error("Return to the top view before marking the remaining unreviewed faces clean.");return
        if self.mesh is not None:
            self.edit(np.flatnonzero(self.mesh.active & (self.labels==-2)),0)
            self.view.setFocus()

    def fill_plaque(self):
        if self.is_read_only() or self.mesh is None or self.labels is None:return
        if self.view.top_only and self.view.side_alignment()<=.08:
            self.show_error('Return to the top view before marking the entire ROI plaque.');return
        self.view.painting=False;self.view.previous=None
        self.end_stroke()
        self.edit(np.flatnonzero(self.mesh.active),1)
        self.view.setFocus()

    def undo(self):
        if self.is_read_only():return
        if self.undo_stack:
            change=self.undo_stack.pop(); ids,before,after=change
            self.labels[ids]=before; self.redo_stack.append(change)
            self.completed=False; self.dirty=True; self.refresh(); self.autosave()

    def redo(self):
        if self.is_read_only():return
        if self.redo_stack:
            change=self.redo_stack.pop(); ids,before,after=change
            self.labels[ids]=after; self.undo_stack.append(change)
            self.completed=False; self.dirty=True; self.refresh(); self.autosave()

    def note_changed(self):
        if self.mesh is not None and not self.is_read_only():
            self.dirty=True

    def refresh(self,*args):
        self.view.overlay_enabled=self.show_overlay.isChecked(); self.view.overlay_alpha=self.opacity.value()/100
        if self.mesh is None:
            return
        self.view.update_labels(self.labels)
        self.label_status.setText('Label: '+('approved' if self.completed else 'pending'))
        self.statusBar().showMessage(self.last_save+('  •  Unsaved edits' if self.dirty else ''))

    def refresh_queue(self):
        if self.store is None:
            return
        done=0;approved=0
        for i,p in enumerate(self.queue):
            status=self.store.status(p); done+=status=='done'
            roi_status=(read_state(self.store,p) or {}).get('status','pending')
            approved+=roi_status=='approved'
            self.list.item(i).setText(f'[ROI {roi_status.upper()} | {status.upper()}] {p.relative_to(self.root)}')
        self.progress.setText(f'ROI: {approved}/{len(self.queue)} approved  •  Plaque: {done}/{len(self.queue)} completed  •  {self.store.annotator}')

    def save(self,*args):
        if self.is_read_only():return True
        if self.mesh is None:
            return True
        self.end_stroke()
        try:
            if current_version(self.store,self.mesh.path)!=self.mesh.review_version:
                raise ValueError('The shared ROI changed. Press F5 to reload the latest shared ROI before saving labels.')
            preview=self.store.path_for(self.mesh.path).parent/(self.mesh.path.name+'.label_colors.ply')
            export_label_colors(self.mesh,self.labels,preview)
            if self.auto_ply.isChecked():
                target=self.store.path_for(self.mesh.path).parent/(self.mesh.path.name+'.labeled.ply')
                export_ply(self.mesh,self.labels,target)
            self.store.save(self.mesh,self.labels,self.completed,self.notes.toPlainText(),self.view.camera_state())
            self.dirty=False; self.last_save='Saved '+QtCore.QTime.currentTime().toString('HH:mm:ss')
            self.refresh_queue(); self.refresh(); return True
        except Exception as exc:
            self.show_error('Saving failed. The current sample has been kept open.\n\n'+str(exc)); return False

    def autosave(self):
        if self.dirty and not self.view.painting:
            self.save()

    def complete_next(self):
        if self.is_read_only():return
        if self.mesh is None:
            return
        self.end_stroke()
        if self.mesh.review_status!='approved':
            self.show_error('Verify and approve the shared specimen ROI before completing plaque annotation.');return
        unreviewed=int(np.sum((self.labels==-2)&self.mesh.active))
        if unreviewed:
            self.show_error(f'{unreviewed:,} faces are still unreviewed. Review the whole ROI, then use “Remaining → clean”, or paint those faces explicitly.')
            return
        if self.mesh.warning:
            self.show_error('The mesh has no verified color display. Fix its prepared texture/RGB data and reopen it before approving labels.'); return
        self.completed=True; self.dirty=True
        if self.save():
            self.next_unfinished()

    def next_unfinished(self):
        if not self.queue:
            return
        order=list(range(self.index+1,len(self.queue)))+list(range(0,self.index))
        target=next((i for i in order if self.store.status(self.queue[i]) not in {'done','error'}),None)
        if target is not None:
            self.list.setCurrentRow(target)
        else:
            self.statusBar().showMessage('No other unfinished samples. You can reopen any sample from the list.')

    def navigate_sample(self,step):
        if not self.queue or self.index<0:return
        target=self.index+step
        if not 0<=target<len(self.queue):
            self.statusBar().showMessage('First sample in this queue.' if step<0 else 'Last sample in this queue.')
            return
        if self.view.painting:
            self.view.painting=False;self.view.previous=None;self.end_stroke()
        # change_sample saves first and restores the selection if saving fails.
        self.list.setCurrentRow(target);self.view.setFocus()

    def choose_texture(self):
        if self.is_read_only():return
        if self.mesh is None:
            return
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'Select the matching mesh texture',str(self.mesh.path.parent),'Images (*.jpg *.jpeg *.png *.tif *.tiff)')
        if path:
            try:
                self.view.set_texture(path); self.notice.setText('Texture loaded • original geometry preserved')
                self.dirty=True; self.autosave()
            except Exception as exc:
                self.show_error(str(exc))

    def export(self):
        if self.mesh is None:
            return
        default=self.store.path_for(self.mesh.path).parent/(self.mesh.path.name+'.labeled.ply')
        path,chosen=QtWidgets.QFileDialog.getSaveFileName(self,'Export labeled mesh',str(default),
            'PLY binary (*.ply);;PLY ASCII (*.ply);;PLY label colors (*.ply);;VTK surface (*.vtp)')
        if path:
            try:
                target=Path(path)
                if not target.suffix:
                    target=target.with_suffix('.vtp' if chosen.startswith('VTK') else '.ply')
                if chosen.startswith('PLY label colors'):
                    if target.name.endswith('.labeled.ply'):target=target.with_name(target.name[:-len('.labeled.ply')]+'.label_colors.ply')
                    export_label_colors(self.mesh,self.labels,target)
                elif target.suffix.lower()=='.ply':export_ply(self.mesh,self.labels,target,binary='ASCII' not in chosen)
                elif target.suffix.lower()=='.vtp':export_vtp(self.mesh,self.labels,target)
                else:raise ValueError('Choose a .ply or .vtp export filename.')
                self.statusBar().showMessage('Exported '+str(target))
            except Exception as exc:
                self.show_error(str(exc))

    def closeEvent(self,event):
        if self.save():
            QtWidgets.QApplication.instance().removeEventFilter(self)
            self.timer.stop(); self.release_lock(); self.view.Finalize(); event.accept()
        else:
            event.ignore()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',help='Prepared ROI dataset folder')
    parser.add_argument('--output',help='Annotation output directory; default ROOT/annotations')
    parser.add_argument('--annotator',help='Rater ID, e.g. johan or rater2')
    parser.add_argument('--manifest',help='Optional preparation annotation_inventory.csv to load only approved scans')
    parser.add_argument('--lock-annotator',action='store_true',help='Disable rater switching during independent study labeling')
    parser.add_argument('--review-mode',action='store_true',help='Other raters are read-only while editing the initial consensus rater')
    parser.add_argument('--demo',action='store_true',help='Generate and open three synthetic disk samples')
    args=parser.parse_args()
    app=QtWidgets.QApplication(sys.argv); app.setApplicationName('BioCal3D Annotator')
    window=Annotator(args); window.show(); sys.exit(app.exec())

if __name__=='__main__':
    main()
