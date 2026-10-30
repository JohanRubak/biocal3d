"""Relocation, identity, checksum validation and machine-local folder memory."""
import argparse,hashlib,importlib,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from original_models import OriginalModelIndex,OriginalModelLookupError,folder_setting_key
from mesh_io import Mesh,polydata,export_ply,atomic_npz,load_mesh
from roi_review_store import load_review_context

class OriginalIndexTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)/'02-09-2026-Data collected';self.root.mkdir()
    def tearDown(self):self.tmp.cleanup()
    def file(self,relative,data=b'correct original'):
        p=self.root/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);return p
    def test_rebased_windows_path_without_tree_walk(self):
        expected=self.file('iTERO/G1/BCG1T0-1/mesh.ply');index=OriginalModelIndex(self.root)
        with patch.object(index,'_scan',side_effect=AssertionError('unnecessary tree walk')):
            found=index.resolve([r'C:\Users\old\data\02-09-2026-Data collected\iTERO\G1\BCG1T0-1\mesh.ply'],scanner='iTERO',sample='BCG1T0-1')
        self.assertEqual(found,expected)
    def test_reorganized_scanner_and_sample_are_exact(self):
        expected=self.file('new/TRIOS5/BCG3T1-1/renamed.ply')
        self.file('new/TRIOS3/BCG3T1-1/renamed.ply');self.file('new/TRIOS5/BCG3T1-10/renamed.ply')
        self.assertEqual(OriginalModelIndex(self.root).resolve(['old/TRIOS5/BCG3T1-1/raw.ply'],scanner='TRIOS5',sample='BCG3T1-1'),expected)
    def test_checksum_disambiguates_and_replacement_invalidates_digest(self):
        good=self.file('copyA/iTERO/BCG1T0-1/mesh.ply',b'good');self.file('copyB/iTERO/BCG1T0-1/mesh.ply',b'bad')
        checksum=hashlib.sha256(b'good').hexdigest();index=OriginalModelIndex(self.root)
        self.assertEqual(index.resolve(['mesh.ply'],scanner='iTERO',sample='BCG1T0-1',checksum=checksum),good)
        good.write_bytes(b'changed')
        with self.assertRaisesRegex(OriginalModelLookupError,'No corresponding'):index.resolve(['mesh.ply'],scanner='iTERO',sample='BCG1T0-1',checksum=checksum)
    def test_duplicate_matches_are_not_chosen(self):
        self.file('copyA/iTERO/BCG1T0-1/mesh.ply');self.file('copyB/iTERO/BCG1T0-1/mesh.ply')
        with self.assertRaisesRegex(OriginalModelLookupError,'Multiple matching'):OriginalModelIndex(self.root).resolve(['mesh.ply'],scanner='iTERO',sample='BCG1T0-1')
    def test_generated_snapshots_are_excluded(self):
        self.file('_development_20261020/iTERO/BCG1T0-1/mesh.ply');self.file('Approved ROIs/samples/iTERO/BCG1T0-1/mesh.ply')
        (self.root/'Approved ROIs/geometry_rebuild.json').write_text('{}')
        expected=self.file('raw/iTERO/BCG1T0-1/mesh.ply')
        self.assertEqual(OriginalModelIndex(self.root).resolve(['mesh.ply'],scanner='iTERO',sample='BCG1T0-1'),expected)
    def test_index_reused_and_refreshes_for_new_file(self):
        first=self.file('new/TRIOS5/BCG3T1-1/renamed.ply');index=OriginalModelIndex(self.root)
        index.resolve(['raw.ply'],scanner='TRIOS5',sample='BCG3T1-1')
        second=self.file('new/TRIOS5/BCG3T1-2/renamed.ply')
        self.assertEqual(index.resolve(['raw.ply'],scanner='TRIOS5',sample='BCG3T1-2'),second)
        with patch.object(index,'_scan',side_effect=AssertionError('index not reused')):
            self.assertEqual(index.resolve(['raw.ply'],scanner='TRIOS5',sample='BCG3T1-1'),first)

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6 import QtCore,QtWidgets
from biocal3d_annotator import Annotator
from roi_review_ui import ReviewDialog
from test_roi_ui import NoRenderWindow,NoRenderInteractor

class OriginalFolderUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    def setUp(self):
        self.q=importlib.import_module('vtkmodules.qt.QVTKRenderWindowInteractor');self.old=self.q.vtkRenderWindow,self.q.vtkGenericRenderWindowInteractor
        self.q.vtkRenderWindow=NoRenderWindow;self.q.vtkGenericRenderWindowInteractor=NoRenderInteractor
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name)
        self.raw=self.base/'new PC/02-09-2026-Data collected';self.dataset=self.base/'Approved ROIs'
        self.v=np.array([[-1.,-1,0],[1,-1,0],[1,1,0],[-1,1,0],[5,5,-2]],float);self.f=np.array([[0,1,2],[0,2,3],[0,4,1]])
        self.raw_files=[]
        for n in (1,2):
            sid=f'BCG1T0-{n}';raw=self.raw/f'iTERO/G1/{sid}/{sid}.ply';raw.parent.mkdir(parents=True)
            mesh=Mesh(raw,self.v,self.f,polydata(self.v,self.f,np.tile([220,60+n,100],(5,1))),np.ones(3,bool))
            export_ply(mesh,None,raw.with_name('temporary.ply'));raw.with_name('temporary.ply').rename(raw);self.raw_files.append(raw)
            old=rf'C:\Users\old\data\02-09-2026-Data collected\iTERO\G1\{sid}\{sid}.ply';folder=self.dataset/f'samples/iTERO/G1/{sid}'
            checksum=hashlib.sha256(raw.read_bytes()).hexdigest()
            atomic_npz(folder/'geometry.npz',vertices=self.v[:4],faces=self.f[:2],rgb=np.tile([220,60+n,100],(4,1)),raw_triangle_id=np.array([0,1]),raw_vertex_id=np.arange(4),source_ply=np.array(old),source_sha256=np.array(checksum),original_to_canonical=np.eye(4),roi_radius=np.array(2.))
            (folder/'metadata.json').write_text(json.dumps(dict(source=old,source_sha256=checksum,scanner='iTERO',sample_id=sid)))
        self.settings_path=str(self.base/'settings.ini');self.owner=self.new_owner();self.dialogs=[]
    def new_owner(self):
        owner=Annotator(argparse.Namespace(root=None,output=None,annotator='qa',demo=False));owner.settings=QtCore.QSettings(self.settings_path,QtCore.QSettings.Format.IniFormat)
        owner.show_error=lambda text:self.fail(str(text));owner.load_folder(self.dataset);return owner
    def dialog(self):
        dialog=ReviewDialog(self.owner);self.dialogs.append(dialog);return dialog
    def close_dialog(self,dialog):
        dialog.reject();dialog.view.Finalize();dialog.deleteLater();self.dialogs.remove(dialog);self.app.processEvents()
    def close_owner(self):
        self.owner.timer.stop();self.owner.draft_timer.stop();self.owner.release_lock();self.owner.mesh=None;self.owner.close();self.owner.deleteLater();self.app.processEvents()
    def tearDown(self):
        for dialog in list(self.dialogs):self.close_dialog(dialog)
        self.close_owner();self.q.vtkRenderWindow,self.q.vtkGenericRenderWindowInteractor=self.old;self.tmp.cleanup()
    def test_select_once_other_sample_and_restart_automatically_load_original(self):
        dialog=self.dialog();self.assertEqual(dialog.base.context_kind,'local');before=self.owner.mesh.path.read_bytes()
        with patch.object(QtWidgets.QFileDialog,'getExistingDirectory',return_value=str(self.raw)):dialog.locate_folder()
        self.assertEqual(dialog.base.context_kind,'full');self.assertEqual(Path(dialog.raw_path),self.raw_files[0]);np.testing.assert_array_equal(dialog.mask,[True,True,False])
        self.assertEqual(before,self.owner.mesh.path.read_bytes());self.assertFalse(dialog.saved)
        self.assertEqual(self.owner.settings.value(folder_setting_key(self.dataset)),str(self.raw.resolve()))
        self.close_dialog(dialog);self.owner.navigate_sample(1)
        with patch.object(QtWidgets.QFileDialog,'getExistingDirectory',side_effect=AssertionError('asked again')):
            dialog=self.dialog();self.assertEqual(Path(dialog.raw_path),self.raw_files[1]);self.assertEqual(dialog.base.context_kind,'full')
        self.close_dialog(dialog);self.close_owner();self.owner=self.new_owner()
        with patch.object(QtWidgets.QFileDialog,'getExistingDirectory',side_effect=AssertionError('forgot after restart')):
            dialog=self.dialog();self.assertEqual(dialog.base.context_kind,'full')
    def test_missing_remembered_folder_opens_local_crop_with_explanation(self):
        self.owner.settings.setValue(folder_setting_key(self.dataset),str(self.base/'missing disk'))
        dialog=self.dialog();self.assertEqual(dialog.base.context_kind,'local');self.assertIn('unavailable',dialog.message.text())
    def test_wrong_checksum_preserves_context_and_does_not_remember_bad_root(self):
        self.raw_files[0].write_bytes(b'wrong model');dialog=self.dialog();before=dialog.base;errors=[];dialog.error=lambda error:errors.append(str(error))
        with patch.object(QtWidgets.QFileDialog,'getExistingDirectory',return_value=str(self.raw)):dialog.locate_folder()
        self.assertTrue(errors);self.assertIs(dialog.base,before);self.assertFalse(self.owner.settings.contains(folder_setting_key(self.dataset)))
    def test_checksum_match_still_requires_geometry_correspondence(self):
        path=self.owner.mesh.path
        with np.load(path,allow_pickle=False) as data:arrays={k:np.array(data[k]) for k in data.files}
        arrays['vertices']=arrays['vertices']+[0,0,.1];atomic_npz(path,**arrays)
        with self.assertRaisesRegex(ValueError,'triangle mapping does not match'):load_review_context(self.owner.store,load_mesh(path),raw_root=self.raw)

if __name__=='__main__':unittest.main()
