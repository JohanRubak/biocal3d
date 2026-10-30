"""Qt control/state smoke test with rendering disabled; picking is tested separately."""
import argparse,importlib,tempfile,unittest
from pathlib import Path
import vtk
from PySide6 import QtWidgets
from biocal3d_annotator import Annotator
from mesh_io import create_demo
from roi_review_ui import ReviewDialog

class NoRenderWindow(vtk.vtkRenderWindow):
    def SetWindowInfo(self,*args):pass
    def Render(self):pass
class NoRenderInteractor(vtk.vtkGenericRenderWindowInteractor):
    def __init__(self):
        super().__init__();self.EnableRenderOff()
    def Initialize(self):pass
    def Render(self):pass

class ReviewUITests(unittest.TestCase):
    def test_controls_commit_reload_and_navigation(self):
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        q=importlib.import_module('vtkmodules.qt.QVTKRenderWindowInteractor')
        oldrw,oldiren=q.vtkRenderWindow,q.vtkGenericRenderWindowInteractor
        q.vtkRenderWindow=NoRenderWindow;q.vtkGenericRenderWindowInteractor=NoRenderInteractor
        vtk.vtkObject.GlobalWarningDisplayOff()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                owner=Annotator(argparse.Namespace(root=None,output=None,annotator='qa',demo=False))
                errors=[];owner.show_error=errors.append
                owner.load_folder(create_demo(Path(tmp)/'demo'));owner.labels[:]=0;owner.dirty=True
                self.assertTrue(owner.save())
                r=ReviewDialog(owner);r.error=errors.append
                r.start_stroke();r.paint_faces([0,1]);r.end_stroke();r.undo_edit();r.redo_edit();r.reset_mask()
                r.original.setChecked(False);r.original.setChecked(True);r.guides.setChecked(False);r.guides.setChecked(True)
                r.controls['below'].setValue(.5);r.automatic();r.commit('approved');self.assertTrue(r.saved,errors)
                owner.reload_shared_roi();self.assertEqual(owner.mesh.review_status,'approved');self.assertFalse(errors)
                self.assertTrue(owner.save());owner.change_sample(1);owner.change_sample(0);self.assertFalse(errors,errors)
                r.deleteLater();owner.close();app.processEvents()
        finally:
            q.vtkRenderWindow,q.vtkGenericRenderWindowInteractor=oldrw,oldiren
            vtk.vtkObject.GlobalWarningDisplayOn()

if __name__=='__main__':unittest.main()
