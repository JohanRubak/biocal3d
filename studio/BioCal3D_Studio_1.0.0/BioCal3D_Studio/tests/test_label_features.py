"""Label preview and reset/save regression tests; no OpenGL display needed."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
from plyfile import PlyData
from mesh_io import atomic_npz, load_mesh, AnnotationStore, PALETTE, export_label_colors, discover
from biocal3d_annotator import Annotator


class HeadlessEditor:
    # Exercise actual application methods with rendering replaced by a no-op.
    clear_labels = Annotator.clear_labels
    edit = Annotator.edit
    save = Annotator.save
    autosave = Annotator.autosave
    end_stroke = Annotator.end_stroke
    undo = Annotator.undo
    redo = Annotator.redo
    def __init__(self, mesh, store):
        self.mesh=mesh; self.store=store
        self.labels=np.array([1,0,-1,-1],np.int8)
        self.completed=True; self.dirty=True; self.stroke={}
        self.undo_stack=[];self.redo_stack=[];self.readonly=False;self.errors=[]
        self.exports_pending=False;self.draft_timer=SimpleNamespace(stop=lambda:None,start=lambda:None)
        self.view=SimpleNamespace(painting=False,previous=None,setFocus=lambda:None,camera_state=lambda:{})
        self.notes=SimpleNamespace(toPlainText=lambda:'keep these notes')
        self.auto_ply=SimpleNamespace(isChecked=lambda:True)
    def is_read_only(self):return self.readonly
    def refresh(self,*args,**kwargs):pass
    def refresh_status(self):pass
    def refresh_queue(self,*args,**kwargs):pass
    def show_error(self,message):self.errors.append(message)


class LabelFeatureTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.path=self.root/'sample/geometry.npz'
        self.vertices=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
        self.faces=np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])
        atomic_npz(self.path,vertices=self.vertices,faces=self.faces,
                   rgb=np.tile([200,90,80],(4,1)),face_roi_mask=[True,True,True,False])
        self.mesh=load_mesh(self.path)
        self.store=AnnotationStore(self.root,self.root/'annotations','rater1')
    def tearDown(self):self.tmp.cleanup()
    def test_color_preview_keeps_face_labels_geometry_and_source_mapping(self):
        labels=np.array([-2,-1,0,1],np.int8)
        before=self.path.read_bytes()
        for binary in (True,False):
            target=self.root/f'preview_{binary}.label_colors.ply'
            export_label_colors(self.mesh,labels,target,binary=binary)
            p=PlyData.read(target);v=p['vertex'].data;f=p['face'].data
            np.testing.assert_array_equal(f['face_label'],labels)
            np.testing.assert_array_equal(f['quality'],labels)
            np.testing.assert_array_equal(f['roi_active'],self.mesh.active)
            np.testing.assert_array_equal(f['source_face_id'],np.arange(4))
            np.testing.assert_array_equal(v['source_vertex_id'],self.faces.reshape(-1))
            coords=np.column_stack([v[n] for n in ('x','y','z')])
            np.testing.assert_array_equal(coords[np.stack(f['vertex_indices'])],self.vertices[self.faces])
            colors=np.column_stack([v[n] for n in ('red','green','blue')])
            np.testing.assert_array_equal(colors,np.repeat([PALETTE[int(x)] for x in labels],3,axis=0))
            self.assertFalse(any(c.startswith('TextureFile') for c in p.comments))
        self.assertEqual(self.path.read_bytes(),before)
        self.assertEqual(discover(self.root),[self.path])
    def test_preview_rejects_invalid_labels_and_input_overwrite(self):
        before=self.path.read_bytes()
        with self.assertRaises(ValueError):export_label_colors(self.mesh,[1],self.root/'bad.ply')
        with self.assertRaises(ValueError):export_label_colors(self.mesh,[1,0,-1,9],self.root/'bad.ply')
        with self.assertRaises(ValueError):export_label_colors(self.mesh,[1,0,-1,-1],self.path)
        self.assertEqual(self.path.read_bytes(),before)
    def test_reset_autosaves_and_undo_redo_preserve_outside_roi(self):
        editor=HeadlessEditor(self.mesh,self.store)
        original=editor.labels.copy();source_bytes=self.path.read_bytes()
        self.assertTrue(editor.save());self.assertEqual(self.store.status(self.path),'done')
        editor.clear_labels()
        np.testing.assert_array_equal(editor.labels,[-2,-2,-2,-1])
        self.assertFalse(editor.completed);self.assertEqual(self.store.status(self.path),'draft')
        loaded,done,notes=self.store.load(self.mesh)
        np.testing.assert_array_equal(loaded,editor.labels);self.assertFalse(done)
        self.assertEqual(notes,'keep these notes')
        editor.undo();np.testing.assert_array_equal(editor.labels,original)
        np.testing.assert_array_equal(self.store.load(self.mesh)[0],original)
        editor.redo();np.testing.assert_array_equal(editor.labels,[-2,-2,-2,-1])
        self.assertEqual(self.path.read_bytes(),source_bytes);self.assertFalse(editor.errors,editor.errors)
        editor.save()
        preview=self.store.path_for(self.path).parent/(self.path.name+'.label_colors.ply')
        np.testing.assert_array_equal(PlyData.read(preview)['face']['face_label'],editor.labels)
    def test_readonly_reset_does_nothing(self):
        editor=HeadlessEditor(self.mesh,self.store);editor.readonly=True
        original=editor.labels.copy();editor.clear_labels()
        np.testing.assert_array_equal(editor.labels,original)
        self.assertTrue(editor.completed);self.assertFalse(editor.undo_stack)
        self.assertFalse(self.store.path_for(self.path).exists())
    def test_colored_save_even_when_original_ply_export_disabled(self):
        editor=HeadlessEditor(self.mesh,self.store)
        editor.auto_ply=SimpleNamespace(isChecked=lambda:False)
        self.assertTrue(editor.save())
        out=self.store.path_for(self.path).parent
        self.assertTrue((out/(self.path.name+'.label_colors.ply')).is_file())
        self.assertTrue(self.store.path_for(self.path).is_file())
        self.assertFalse((out/(self.path.name+'.labeled.ply')).exists())


if __name__=='__main__':unittest.main()
