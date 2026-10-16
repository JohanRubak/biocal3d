"""Approve displayed geometry, independent review navigation and default modes."""
import argparse,importlib,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import vtk
from PySide6 import QtCore,QtWidgets,QtTest
from biocal3d_annotator import Annotator
from mesh_io import atomic_npz,load_mesh,mesh_appearance
from roi_review_store import (approve_displayed_surface,read_state,read_reviewed,
                              load_review_context,save_review,current_version)
from roi_review_ui import ReviewDialog
from test_roi_ui import NoRenderWindow,NoRenderInteractor


class QuickReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.q=importlib.import_module('vtkmodules.qt.QVTKRenderWindowInteractor')
        self.previous=self.q.vtkRenderWindow,self.q.vtkGenericRenderWindowInteractor
        self.q.vtkRenderWindow=NoRenderWindow;self.q.vtkGenericRenderWindowInteractor=NoRenderInteractor
        vtk.vtkObject.GlobalWarningDisplayOff()
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.v=np.array([[-1.,-1,0],[1,-1,0],[1,1,0],[-1,1,0],[9,9,9]])
        self.f=np.array([[0,1,2],[0,2,3]])
        for name in ('a','b','c'):
            atomic_npz(self.root/name/'geometry.npz',vertices=self.v,faces=self.f,
                       rgb=np.tile([220,80,100],(len(self.v),1)))
        self.owner=Annotator(argparse.Namespace(root=None,output=None,annotator='qa',demo=False))
        self.errors=[];self.owner.show_error=self.errors.append;self.owner.load_folder(self.root)

    def tearDown(self):
        self.owner.timer.stop();self.owner.release_lock();self.owner.mesh=None;self.owner.dirty=False
        self.owner.close();self.owner.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None,QtCore.QEvent.Type.DeferredDelete)
        self.app.processEvents()
        self.q.vtkRenderWindow,self.q.vtkGenericRenderWindowInteractor=self.previous
        vtk.vtkObject.GlobalWarningDisplayOn();self.tmp.cleanup()

    def test_button_approves_exact_layout_preserves_labels_and_skips_approved_roi(self):
        w=self.owner;first=w.mesh.path;before=w.mesh.fingerprint;source_bytes=first.read_bytes()
        approve_displayed_surface(w.store,read_reviewed(w.store,w.queue[1]))
        w.labels[:]=[1,-2];w.notes.setPlainText('Existing plaque notes');w.dirty=True
        w.approve_roi_button.click()
        self.assertFalse(self.errors,self.errors);self.assertEqual(w.index,2)
        state=read_state(w.store,first);self.assertEqual(state['status'],'approved')
        self.assertEqual(state['review_action'],'approve_existing_geometry')
        mesh=read_reviewed(w.store,first);self.assertEqual(mesh.fingerprint,before)
        np.testing.assert_array_equal(mesh.vertices,self.v);np.testing.assert_array_equal(mesh.faces,self.f)
        self.assertEqual(first.read_bytes(),source_bytes)
        labels,done,notes=w.store.load(mesh)
        np.testing.assert_array_equal(labels,[1,-2]);self.assertFalse(done);self.assertEqual(notes,'Existing plaque notes')
        self.assertIn('ROI: 2/3 approved',w.progress.text())
        w.approve_roi_button.click();self.assertFalse(self.errors,self.errors)
        self.assertIn('All ROIs in this queue are approved',w.statusBar().currentMessage())
        version=w.mesh.review_version;w.approve_roi_button.click()
        self.assertEqual(w.mesh.review_version,version);self.assertFalse(w.completed)

    def test_failure_keeps_current_sample_and_readonly_cannot_approve(self):
        w=self.owner;first=w.mesh.path
        with patch('biocal3d_annotator.approve_displayed_surface',side_effect=RuntimeError('simulated conflict')):
            w.approve_roi_button.click()
        self.assertEqual(w.index,0);self.assertIsNone(read_state(w.store,first))
        self.assertIn('simulated conflict',self.errors[-1])
        w.review_owner='other';w.update_review_mode()
        self.assertFalse(w.approve_roi_button.isEnabled());w.approve_roi_next()
        self.assertIsNone(read_state(w.store,first));self.assertEqual(w.index,0)

    def test_stale_opened_surface_cannot_approve_a_newer_geometry(self):
        w=self.owner;stale=w.mesh
        base,mask,s,_,raw=load_review_context(w.store,stale)
        mask[1]=False;save_review(w.store,stale.path,base,mask,s,'pending','trimmed','',raw)
        with self.assertRaises(RuntimeError):approve_displayed_surface(w.store,stale)
        current=read_reviewed(w.store,stale.path)
        record=approve_displayed_surface(w.store,current)
        result=read_reviewed(w.store,stale.path)
        self.assertEqual(record['notes'],'trimmed');self.assertEqual(result.fingerprint,current.fingerprint)
        self.assertEqual(len(result.faces),1);self.assertEqual(result.review_status,'approved')

    def test_defaults_show_context_live_and_restore_without_reextracting_on_open(self):
        r=ReviewDialog(self.owner)
        try:
            before=r.mask.copy()
            self.assertTrue(r.original.isChecked());self.assertTrue(r.live.isChecked())
            self.assertEqual(r.operation.currentIndex(),1)
            self.assertFalse(r.timer.isActive())
            QtTest.QTest.qWait(260);np.testing.assert_array_equal(r.mask,before)
            r.controls['z'].setValue(2);QtTest.QTest.qWait(260)
            self.assertFalse(r.mask.any())
        finally:r.reject();r.deleteLater()

    def test_keyboard_approval_advances_once(self):
        w=self.owner;w.show();w.activateWindow();w.view.setFocus();self.app.processEvents()
        QtTest.QTest.keyClick(w.view,QtCore.Qt.Key.Key_Return,QtCore.Qt.KeyboardModifier.ControlModifier)
        self.assertEqual(w.index,1);self.assertEqual(read_state(w.store,w.queue[0])['status'],'approved')
        self.assertIsNone(read_state(w.store,w.queue[1]));self.assertFalse(self.errors,self.errors)

    def test_approval_preserves_inactive_mask_and_exact_textured_geometry(self):
        from mesh_io import Mesh,polydata,export_ply
        from PIL import Image
        w=self.owner;source=self.root/'textured.ply'
        uv=np.array([[0,0],[1,0],[1,1],[.2,0],[1,1],[0,1]],float)
        texture=self.root/'texture.png';Image.fromarray(np.zeros((4,4,3),np.uint8)).save(texture)
        m=Mesh(self.root/'input.ply',self.v,self.f,polydata(self.v[self.f.ravel()],np.arange(6).reshape(-1,3),None,uv),np.ones(2,bool),texture)
        export_ply(m,None,source,face_ids=np.array([4,8]))
        m=load_mesh(source);before=m.fingerprint;before_uv=mesh_appearance(m)[1].copy()
        approve_displayed_surface(w.store,m);result=read_reviewed(w.store,source)
        self.assertEqual(result.fingerprint,before)
        np.testing.assert_array_equal(mesh_appearance(result)[1],before_uv)
        np.testing.assert_array_equal(result.source_face_ids,[4,8]);self.assertEqual(result.texture.read_bytes(),texture.read_bytes())
        source=self.root/'inactive/geometry.npz'
        atomic_npz(source,vertices=self.v,faces=self.f,face_roi_mask=np.array([True,False]),rgb=np.tile([220,80,100],(len(self.v),1)))
        m=load_mesh(source);approve_displayed_surface(w.store,m)
        result=read_reviewed(w.store,source)
        np.testing.assert_array_equal(result.active,[True,False]);self.assertEqual(result.fingerprint,m.fingerprint)

    def test_review_version_and_existing_completed_labels_survive_approval(self):
        w=self.owner;w.labels[:]=[1,0];w.completed=True;w.dirty=True
        self.assertTrue(w.save());w.approve_roi_next()
        first=w.queue[0];labels,done,_=w.store.load(read_reviewed(w.store,first))
        np.testing.assert_array_equal(labels,[1,0]);self.assertTrue(done)
        self.assertFalse(self.errors,self.errors)


if __name__=='__main__':unittest.main()
