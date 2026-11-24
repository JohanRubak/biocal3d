"""Performance regressions with data-integrity and safe-save checks."""
import argparse,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from mesh_io import AnnotationStore,atomic_npz,load_mesh,export_ply,mesh_appearance
from roi_review_store import read_reviewed,resolve_review_source,approve_displayed_surface
from performance import MeshCache

class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.path=self.root/'sample/geometry.npz'
        self.v=np.array([[-1.,-1,0],[1,-1,0],[1,1,0],[-1,1,0]])
        self.f=np.array([[0,1,2],[0,2,3]])
        atomic_npz(self.path,vertices=self.v,faces=self.f,rgb=np.tile([220,60,100],(4,1)))
        self.store=AnnotationStore(self.root,self.root/'annotations','qa');self.store.mesh_cache=MeshCache()
        approve_displayed_surface(self.store,read_reviewed(self.store,self.path))
    def tearDown(self):self.tmp.cleanup()
    def test_revisit_uses_cache_but_labels_are_fresh_and_profiles_independent(self):
        a=read_reviewed(self.store,self.path)
        with patch('roi_review_store.load_mesh',side_effect=AssertionError('cache missed')):
            b=read_reviewed(self.store,self.path)
        self.assertIsNot(a,b);self.assertEqual(a.fingerprint,b.fingerprint)
        self.store.save(a,np.array([1,0],np.int8),True,'first')
        np.testing.assert_array_equal(self.store.load(b)[0],[1,0])
        self.store.save(a,np.array([0,1],np.int8),False,'changed')
        np.testing.assert_array_equal(self.store.load(b)[0],[0,1])
        other=AnnotationStore(self.root,self.root/'annotations','other');other.mesh_cache=self.store.mesh_cache
        np.testing.assert_array_equal(other.load(read_reviewed(other,self.path))[0],[-2,-2])
        with self.assertRaises(ValueError):b.vertices[0,0]=99
    def test_changed_input_is_rejected_even_after_cache_hit(self):
        read_reviewed(self.store,self.path)
        atomic_npz(self.path,vertices=self.v+[0,0,.1],faces=self.f,rgb=np.tile([220,60,100],(4,1)))
        with self.assertRaisesRegex(ValueError,'original ROI changed'):read_reviewed(self.store,self.path)
    def test_new_review_invalidates_cache_and_resets_completion(self):
        from roi_review_store import load_review_context,save_review
        m=read_reviewed(self.store,self.path);self.store.save(m,np.array([1,0],np.int8),True,'old')
        base,mask,settings,_,raw=load_review_context(self.store,m);mask[1]=False
        rec=save_review(self.store,self.path,base,mask,settings,'approved','trim',m.review_version,raw)
        new=read_reviewed(self.store,self.path)
        self.assertEqual(len(new.faces),1);self.assertEqual(new.review_version,rec['version'])
        labels,completed,_=self.store.load(new);self.assertFalse(completed);np.testing.assert_array_equal(labels,[1])
    def test_cache_is_bounded(self):
        cache=MeshCache(max_items=1);self.store.mesh_cache=cache
        first=read_reviewed(self.store,self.path)
        p=self.root/'other/geometry.npz';atomic_npz(p,vertices=self.v,faces=self.f,rgb=np.tile([50,100,200],(4,1)))
        second=read_reviewed(self.store,p)
        self.assertEqual(len(cache.entries),1);self.assertEqual(first.fingerprint,second.fingerprint)
        tiny=MeshCache(max_bytes=1);tiny.put('x',(),first);self.assertFalse(tiny.entries)
    def test_lazy_sibling_selection_still_verifies_before_opening(self):
        ply=self.path.with_name('roi.ply');m=load_mesh(self.path);export_ply(m,None,ply)
        # Review lives under geometry.npz. Discovery must not parse every mesh.
        store=AnnotationStore(self.root,self.root/'annotations','qa');store.mesh_cache=MeshCache()
        with patch('roi_review_store.load_mesh',side_effect=AssertionError('eager discovery')):
            self.assertEqual(resolve_review_source(store,ply,verify_geometry=False),self.path)
        m.vertices=m.vertices+[0,0,.1];export_ply(m,None,ply)
        with self.assertRaisesRegex(ValueError,'different input geometry'):read_reviewed(store,self.path)

# Reuse the existing Qt lifecycle and nonrendering viewport fixture, not its tests.
from test_quick_roi_review import QuickReviewTests as Fixture
from PySide6 import QtWidgets
class SavePerformanceTests(unittest.TestCase):
    setUpClass=classmethod(lambda cls:setattr(cls,'app',QtWidgets.QApplication.instance() or QtWidgets.QApplication([])))
    setUp=Fixture.setUp
    tearDown=Fixture.tearDown
    def test_brush_autosave_persists_npz_without_mesh_exports_and_explicit_save_flushes(self):
        w=self.owner;w.edit(np.array([0]),1)
        np.testing.assert_array_equal(w.store.load(w.mesh)[0],[1,-2])
        preview=w.store.path_for(w.mesh.path).parent/(w.mesh.path.name+'.label_colors.ply')
        self.assertFalse(preview.exists());self.assertTrue(w.exports_pending)
        self.assertTrue(w.save());self.assertTrue(preview.exists());self.assertFalse(w.exports_pending)
        from plyfile import PlyData
        np.testing.assert_array_equal(PlyData.read(preview)['face']['face_label'],w.labels)
        self.assertFalse(self.errors,self.errors)
    def test_reopened_draft_flushes_stale_ply_even_after_pending_flag_is_lost(self):
        w=self.owner;w.save();w.edit(np.array([0]),1)
        w.exports_pending=False  # Simulate reopening after a crash.
        self.assertTrue(w.save())
        from plyfile import PlyData
        out=w.store.path_for(w.mesh.path).parent/(w.mesh.path.name+'.label_colors.ply')
        np.testing.assert_array_equal(PlyData.read(out)['face']['face_label'],w.labels)
        self.assertFalse(self.errors,self.errors)
    def test_unchanged_navigation_skips_save_and_export(self):
        w=self.owner;self.assertTrue(w.save())
        source=w.store.path_for(w.mesh.path);before=source.stat().st_mtime_ns
        with patch('biocal3d_annotator.export_label_colors',side_effect=AssertionError('redundant export')):
            w.navigate_sample(1)
        self.assertEqual(w.index,1);self.assertEqual(before,source.stat().st_mtime_ns);self.assertFalse(self.errors,self.errors)
    def test_local_save_does_not_scan_every_annotation(self):
        w=self.owner
        with patch.object(w.store,'status',wraps=w.store.status) as status:
            w.edit(np.array([0]),1)
            self.assertEqual(status.call_count,1)
        self.assertFalse(self.errors,self.errors)
    def test_save_failure_keeps_work_and_blocks_navigation(self):
        w=self.owner;w.edit(np.array([0]),1);index=w.index;labels=w.labels.copy()
        with patch.object(w.store,'save',side_effect=OSError('disk unavailable')):
            w.dirty=True;w.navigate_sample(1)
        self.assertEqual(w.index,index);np.testing.assert_array_equal(w.labels,labels)
        self.assertTrue(w.dirty);self.assertTrue(self.errors)
    def test_overlay_delta_matches_full_rebuild_without_extra_renders(self):
        w=self.owner;w.refresh();w.view.overlay_alpha=.6;w.refresh()
        w.start_stroke();w.paint_faces(np.array([0]));w.end_stroke()
        delta=w.view._overlay_rgba.copy();w.refresh()
        np.testing.assert_array_equal(delta,w.view._overlay_rgba)
        self.assertTrue(w.draft_timer.isActive());w.start_stroke();self.assertFalse(w.draft_timer.isActive())

del Fixture
if __name__=='__main__':unittest.main()
