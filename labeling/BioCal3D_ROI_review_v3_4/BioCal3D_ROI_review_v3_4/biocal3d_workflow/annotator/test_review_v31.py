"""Regressions for context toggling, depth previews, rotation focus and iTERO UV seams."""
import argparse,importlib,tempfile,types,unittest
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from PySide6 import QtCore,QtGui,QtWidgets,QtTest
from mesh_io import Mesh,polydata,load_mesh,export_ply,atomic_npz,AnnotationStore,mesh_appearance
from source_correspondence import triangle_correspondence
from surface_display import display_subset
from roi_review_store import load_review_context,save_review,read_reviewed
from biocal3d_annotator import SurfaceView,Annotator
from roi_review_ui import ReviewDialog
from test_roi_ui import NoRenderWindow,NoRenderInteractor

class CorrespondenceTests(unittest.TestCase):
    def test_reindexed_corners_and_reordered_faces(self):
        source=np.array([[[0.,0,0],[1,0,0],[0,1,0]],[[2,0,0],[3,0,0],[2,1,0]]])
        target=source[[1,0]][:,[2,0,1]]
        rows,corners=triangle_correspondence(source,target)
        np.testing.assert_array_equal(rows,[1,0])
        np.testing.assert_array_equal(source[rows][np.arange(2)[:,None],corners],target)
    def test_different_mesh_rejected(self):
        source=np.array([[[0.,0,0],[1,0,0],[0,1,0]]])
        with self.assertRaises(ValueError):triangle_correspondence(source,source+.1)
    def test_ambiguous_reordered_faces_rejected(self):
        a=np.array([[0.,0,0],[1,0,0],[0,1,0]]);b=a+3
        with self.assertRaises(ValueError):triangle_correspondence(np.array([a,a,b]),np.array([b,a,a]))
    def test_real_vtk_ply_texture_seam_reindexing(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);v=np.array([[0.,0,0],[1,0,0],[1,1,0],[0,1,0]]);f=np.array([[0,1,2],[0,2,3]])
            uv=np.array([[0,0],[1,0],[1,1],[.2,0],[1,1],[0,1]],float)
            texture=root/'texture.png';Image.fromarray(np.zeros((4,4,3),np.uint8)).save(texture)
            raw=root/'raw.ply';m=Mesh(root/'input',v,f,polydata(v[f.ravel()],np.arange(6).reshape(2,3),None,uv),np.ones(2,bool),texture)
            export_ply(m,None,raw)
            reader=vtk.vtkPLYReader();reader.SetFileName(str(raw));reader.Update()
            data=reader.GetOutput();rv=vtk_to_numpy(data.GetPoints().GetData()).astype(float);rf=vtk_to_numpy(data.GetPolys().GetData()).reshape(-1,4)[:,1:]
            self.assertFalse(np.array_equal(rf,f),'Fixture must reproduce VTK UV vertex splitting')
            prepared=root/'prepared';prepared.mkdir();roi=prepared/'roi.ply'
            m=Mesh(root/'input',rv,rf,polydata(rv,rf,np.tile([220,50,80],(len(rv),1))),np.ones(2,bool))
            export_ply(m,None,roi,face_ids=np.arange(2))
            atomic_npz(prepared/'geometry.npz',vertices=rv,faces=rf,raw_triangle_id=np.arange(2),raw_vertex_id=np.arange(len(rv)),
                       source_ply=np.array(str(raw)),source_texture=np.array(str(texture)),original_to_canonical=np.eye(4),roi_radius=np.array(2.))
            store=AnnotationStore(root,root/'annotations','qa');mesh=load_mesh(roi)
            store.save(mesh,np.array([1,0],np.int8),True,'original labels')
            context,mask,settings,_,path=load_review_context(store,mesh)
            self.assertEqual(context.context_kind,'full');np.testing.assert_array_equal(context.faces,rf)
            from roi_geometry import select_surface
            automatic,metrics=select_surface(context.vertices,context.faces,settings)
            np.testing.assert_array_equal(automatic,mask)
            self.assertEqual(metrics['components'],1)
            np.testing.assert_allclose(mesh_appearance(context)[1],uv.reshape(2,6))
            save_review(store,roi,context,mask,settings,'approved','','',path)
            labels,done,_=store.load(read_reviewed(store,roi));np.testing.assert_array_equal(labels,[1,0])

class HeightDefaultTests(unittest.TestCase):
    def test_half_millimetre_default_includes_boundaries(self):
        from roi_geometry import Extraction,select_surface
        heights=[-.6,-.5,0.,.5,.6];vertices=[];faces=[]
        for z in heights:
            k=len(vertices);vertices.extend([[-1,-1,z],[1,-1,z],[0,1,z]]);faces.append([k,k+1,k+2])
        mask,_=select_surface(np.array(vertices),np.array(faces),Extraction(radius=4,connected=False))
        np.testing.assert_array_equal(mask,[False,True,True,True,False])
    def test_new_review_uses_new_height_default_and_saved_review_is_preserved(self):
        from roi_geometry import Extraction
        from dataclasses import asdict
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'geometry.npz';v=np.array([[0.,0,0],[1,0,0],[0,1,0]]);f=np.array([[0,1,2]])
            settings=Extraction(radius=2,below=1.5,above=1.5)
            atomic_npz(path,vertices=v,faces=f,rgb=np.tile([220,80,100],(3,1)),roi_radius=np.array(2.),
                       extraction_json=np.array(json.dumps({'settings':asdict(settings)})))
            store=AnnotationStore(root,root/'annotations','qa');mesh=load_mesh(path)
            base,mask,s,_,raw=load_review_context(store,mesh)
            self.assertEqual((s.below,s.above),(.5,.5))
            s.below=.7;s.above=.9;save_review(store,path,base,mask,s,'pending','','',raw)
            _,_,reopened,_,_=load_review_context(store,read_reviewed(store,path))
            self.assertEqual((reopened.below,reopened.above),(.7,.9))

class DisplayTests(unittest.TestCase):
    def fixture(self):
        v=np.array([[-2.,-1,0],[-.2,-1,0],[-1,1,0],[.2,-1,0],[2,-1,0],[1,1,0]])
        f=np.array([[0,1,2],[3,4,5]])
        return v,f,polydata(v,f,np.tile([200,80,120],(6,1)))
    def test_explicit_subset_and_texture_arrays(self):
        v,f,p=self.fixture();subset,ids=display_subset(p,[1])
        self.assertEqual(subset.GetNumberOfCells(),1);np.testing.assert_array_equal(ids,[1])
        self.assertIsNotNone(subset.GetPointData().GetScalars())
        empty,ids=display_subset(p,[]);self.assertEqual(empty.GetNumberOfCells(),0)
    def test_rendered_context_toggle_and_source_face_picking(self):
        v,f,p=self.fixture();mesh=Mesh(Path('test.ply'),v,f,p,np.ones(2,bool))
        rw=vtk.vtkRenderWindow();rw.SetOffScreenRendering(1);rw.SetSize(400,400);rw.SetMultiSamples(0)
        renderer=vtk.vtkRenderer();renderer.SetBackground(0,0,0);rw.AddRenderer(renderer)
        mapper=vtk.vtkPolyDataMapper();actor=vtk.vtkActor();actor.SetMapper(mapper);actor.GetProperty().LightingOff();renderer.AddActor(actor)
        subset,ids=display_subset(p,[1]);mapper.SetInputData(subset)
        camera=renderer.GetActiveCamera();camera.SetPosition(0,0,10);camera.SetFocalPoint(0,0,0);camera.SetViewUp(0,1,0);camera.ParallelProjectionOn();camera.SetParallelScale(2.5)
        renderer.ResetCameraClippingRange();rw.Render()
        grab=vtk.vtkWindowToImageFilter();grab.SetInput(rw);grab.ReadFrontBufferOff();grab.Update()
        before=vtk_to_numpy(grab.GetOutput().GetPointData().GetScalars()).copy()
        view=types.SimpleNamespace(mesh=mesh,projection_key=None,projected=None,centers=v[f].mean(axis=1),display_face_ids=ids,
                                   renderer=renderer,base=actor,overlay=vtk.vtkActor(),circle=vtk.vtkActor2D(),GetRenderWindow=lambda:rw)
        view._project_centers=types.MethodType(SurfaceView._project_centers,view)
        locator=vtk.vtkStaticCellLocator();locator.SetDataSet(subset);locator.BuildLocator()
        view.picker=vtk.vtkCellPicker();view.picker.PickFromListOn();view.picker.AddPickList(actor);view.picker.AddLocator(locator)
        xy=view._project_centers()[1];selected=SurfaceView.brush_faces(view,*xy,35)
        np.testing.assert_array_equal(selected,[1])
        full,_=display_subset(p,[0,1]);mapper.SetInputData(full);rw.Render();grab.Modified();grab.Update()
        after=vtk_to_numpy(grab.GetOutput().GetPointData().GetScalars()).copy()
        self.assertGreater(np.count_nonzero(before!=after),100)
        rw.Finalize()

class ReviewControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    def setUp(self):
        self.q=importlib.import_module('vtkmodules.qt.QVTKRenderWindowInteractor')
        self.old=(self.q.vtkRenderWindow,self.q.vtkGenericRenderWindowInteractor)
        self.q.vtkRenderWindow=NoRenderWindow;self.q.vtkGenericRenderWindowInteractor=NoRenderInteractor
        vtk.vtkObject.GlobalWarningDisplayOff()
        self.tmp=tempfile.TemporaryDirectory();root=Path(self.tmp.name)
        v=np.array([[-1.,-1,0],[1,-1,0],[0,1,0],[-1,-1,-.8],[1,-1,-.8],[0,1,-.8]])
        atomic_npz(root/'case/geometry.npz',vertices=v,faces=np.array([[0,1,2],[3,4,5]]),rgb=np.tile([220,80,100],(6,1)))
        self.owner=Annotator(argparse.Namespace(root=None,output=None,annotator='qa',demo=False));self.errors=[];self.owner.show_error=self.errors.append
        self.owner.load_folder(root);self.r=ReviewDialog(self.owner);self.r.error=self.errors.append
        self.r.show();self.r.activateWindow();self.app.processEvents()
    def tearDown(self):
        self.r.reject();self.r.deleteLater();self.owner.close();self.owner.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None,QtCore.QEvent.Type.DeferredDelete)
        self.app.processEvents();self.q.vtkRenderWindow,self.q.vtkGenericRenderWindowInteractor=self.old
        vtk.vtkObject.GlobalWarningDisplayOn();self.tmp.cleanup()
    def configure(self):
<<<<<<< HEAD
        r=self.r;r.connected.setChecked(False);r.controls['z'].setValue(0);r.controls['radius'].setValue(3)
=======
        r=self.r;r.original.setChecked(False);r.operation.setCurrentIndex(0);r.connected.setChecked(False);r.controls['z'].setValue(0);r.controls['radius'].setValue(3)
>>>>>>> 3dc377b2c13f84fb1a8141a5ae83a69198e2addb
        r.controls['above'].setValue(1);r.controls['below'].setValue(1.5);r.automatic()
    def test_window_buttons_and_large_default(self):
        r=self.r
        self.assertTrue(r.windowFlags()&QtCore.Qt.WindowType.WindowMinimizeButtonHint)
        self.assertTrue(r.windowFlags()&QtCore.Qt.WindowType.WindowMaximizeButtonHint)
        self.assertTrue(r.windowState()&QtCore.Qt.WindowState.WindowMaximized)
        self.assertGreaterEqual(r.minimumWidth(),700)
    def test_depth_live_preview_context_toggle_and_empty_selection(self):
        r=self.r;self.configure();self.assertEqual(int(r.mask.sum()),2)
        r.live.setChecked(True);r.controls['below'].setValue(.5);QtTest.QTest.qWait(260)
        np.testing.assert_array_equal(r.mask,[True,False]);self.assertEqual(r.view.mapper.GetInput().GetNumberOfCells(),1)
        self.assertIn('below 0.5',r.preview_status.text())
        self.assertAlmostEqual(r.actors[2].GetMapper().GetInput().GetBounds()[4],-.5,places=5)
        r.original.setChecked(True);self.assertEqual(r.view.mapper.GetInput().GetNumberOfCells(),2)
        r.original.setChecked(False);self.assertEqual(r.view.mapper.GetInput().GetNumberOfCells(),1)
        r.controls['below'].setValue(.4);QtTest.QTest.qWait(260);self.assertIn('unchanged',r.preview_status.text())
        r.controls['z'].setValue(2);QtTest.QTest.qWait(260)
        self.assertFalse(r.mask.any());self.assertEqual(r.view.mapper.GetInput().GetNumberOfCells(),0)
        r.commit('approved');self.assertFalse(r.saved);self.assertIn('empty',str(self.errors[-1]))
    def test_apply_keeps_all_eligible_opened_faces_across_uv_seams_and_steep_detail(self):
        from test_roi_review import SurfaceTests
        from roi_geometry import Extraction
        v,f=SurfaceTests.detail_fixture(self)
        v=v[f.ravel()];f=np.arange(len(v)).reshape(-1,3)
<<<<<<< HEAD
        r=self.r;r.base=Mesh(Path('synthetic.ply'),v,f,polydata(v,f,None),np.ones(len(f),bool))
=======
        r=self.r;r.original.setChecked(False);r.base=Mesh(Path('synthetic.ply'),v,f,polydata(v,f,None),np.ones(len(f),bool))
>>>>>>> 3dc377b2c13f84fb1a8141a5ae83a69198e2addb
        r.base.context_kind='full';r.mask=np.r_[np.ones(32,bool),False,False]
        r.settings=Extraction(radius=2,max_angle=25)
        for name,control in r.controls.items():
            blocker=QtCore.QSignalBlocker(control);control.setValue(getattr(r.settings,name));del blocker
        r.view.set_mesh(r.base);r._visible_request=None;r._applied_settings=vars(r.settings).copy()
        r.undo=[];r.redo=[];before=r.mask.copy()
        for _ in range(3):
            self.assertTrue(r.automatic());np.testing.assert_array_equal(r.mask,before)
        self.assertFalse(r.undo);self.assertFalse(self.errors)
        self.assertEqual(r.view.mapper.GetInput().GetNumberOfCells(),32)

    def test_opened_mismatch_explained_and_repeated_apply_is_stable(self):
        r=self.r
        self.assertIn('Opened saved surface',r.preview_status.text())
        self.assertIn('remove',r.preview_status.text())
        self.configure();r.undo=[];r.redo=[];before=r.mask.copy()
        self.assertTrue(r.automatic());np.testing.assert_array_equal(r.mask,before)
        self.assertFalse(r.undo)
        r.controls['below'].setValue(.5);r.automatic();changed=r.mask.copy()
        self.assertIn('1 by height',r.preview_status.text())
        r.automatic();np.testing.assert_array_equal(r.mask,changed)
        self.assertEqual(len(r.undo),1)
        r.undo_edit();np.testing.assert_array_equal(r.mask,before)

    def test_space_after_spinbox_slider_combo_and_button(self):
        r=self.r
        for widget in (r.controls['below'],r.controls['below'].lineEdit(),r.radius,r.operation,r.paint):
            widget.setFocus();self.app.processEvents()
            QtTest.QTest.keyPress(widget,QtCore.Qt.Key.Key_Space)
            self.assertTrue(r.view.space_down,type(widget).__name__)
            self.assertIs(self.app.focusWidget(),r.view)
            self.assertFalse(r.view.painting)
            QtTest.QTest.keyRelease(r.view,QtCore.Qt.Key.Key_Space);self.assertFalse(r.view.space_down)
    def test_context_toggle_preserves_camera_pose_and_zoom(self):
        r=self.r;self.configure();r.start_stroke();r.paint_faces([1]);r.end_stroke()
        camera=r.view.renderer.GetActiveCamera()
        camera.SetPosition(7,-5,8);camera.SetFocalPoint(.2,.4,-.2);camera.SetViewUp(0,0,1)
        camera.SetParallelScale(2.7);camera.SetViewAngle(37);camera.SetWindowCenter(.12,-.15)
        def pose():
            return (camera.GetPosition(),camera.GetFocalPoint(),camera.GetViewUp(),camera.GetParallelScale(),
                    camera.GetViewAngle(),camera.GetParallelProjection(),camera.GetWindowCenter())
        before=pose()
        for visible in (True,False,True,False):
            r.original.setChecked(visible)
            self.assertEqual(pose(),before)
            self.assertEqual(r.view.mapper.GetInput().GetNumberOfCells(),2 if visible else 1)

    def test_geometry_undo_redo_keys_from_controls_preserve_plaque_labels(self):
        r=self.r;self.configure();r.undo=[];r.redo=[]
        original=r.mask.copy();labels=self.owner.labels.copy()
        r.start_stroke();r.paint_faces([0]);r.end_stroke();before_last=r.mask.copy()
        r.start_stroke();r.paint_faces([1]);r.end_stroke();edited=r.mask.copy()
        r.controls['radius'].lineEdit().setFocus();self.app.processEvents()
        QtTest.QTest.keyClick(r.controls['radius'].lineEdit(),QtCore.Qt.Key.Key_Z,QtCore.Qt.KeyboardModifier.ControlModifier)
        np.testing.assert_array_equal(r.mask,before_last)
        r.operation.setFocus();self.app.processEvents()
        QtTest.QTest.keyClick(r.operation,QtCore.Qt.Key.Key_Y,QtCore.Qt.KeyboardModifier.ControlModifier)
        np.testing.assert_array_equal(r.mask,edited)
        r.view.setFocus();self.app.processEvents()
        QtTest.QTest.keyClick(r.view,QtCore.Qt.Key.Key_Z,QtCore.Qt.KeyboardModifier.ControlModifier)
        np.testing.assert_array_equal(r.mask,before_last)
        QtTest.QTest.keyClick(r.view,QtCore.Qt.Key.Key_Z,QtCore.Qt.KeyboardModifier.ControlModifier|QtCore.Qt.KeyboardModifier.ShiftModifier)
        np.testing.assert_array_equal(r.mask,edited)
        np.testing.assert_array_equal(self.owner.labels,labels)

    def test_undo_redo_restores_extraction_parameters_and_cancels_pending_preview(self):
        r=self.r;self.configure();r.undo=[];r.redo=[];original=r.mask.copy()
        r.live.setChecked(True);r.timer.stop();r.controls['below'].setValue(.5)
        self.assertTrue(r.timer.isActive())
        r.controls['below'].setFocus();self.app.processEvents()
        QtTest.QTest.keyClick(r.controls['below'],QtCore.Qt.Key.Key_Z,QtCore.Qt.KeyboardModifier.ControlModifier)
        self.assertEqual(r.controls['below'].value(),1.5);np.testing.assert_array_equal(r.mask,original)
        QtTest.QTest.qWait(260);np.testing.assert_array_equal(r.mask,original);self.assertFalse(r.timer.isActive())
        QtTest.QTest.keyClick(r.controls['below'],QtCore.Qt.Key.Key_Y,QtCore.Qt.KeyboardModifier.ControlModifier)
        self.assertEqual(r.controls['below'].value(),.5);np.testing.assert_array_equal(r.mask,[True,False])

    def test_notes_keep_typing_spaces(self):
        r=self.r;r.notes.setFocus();QtGui.QCursor.setPos(r.notes.mapToGlobal(QtCore.QPoint(4,4)));self.app.processEvents()
        QtTest.QTest.keyClicks(r.notes,'test note');self.assertEqual(r.notes.toPlainText(),'test note');self.assertFalse(r.view.space_down)
        before=r.mask.copy();QtTest.QTest.keyClick(r.notes,QtCore.Qt.Key.Key_Z,QtCore.Qt.KeyboardModifier.ControlModifier)
        self.assertNotEqual(r.notes.toPlainText(),'test note');np.testing.assert_array_equal(r.mask,before)

if __name__=='__main__':unittest.main()
