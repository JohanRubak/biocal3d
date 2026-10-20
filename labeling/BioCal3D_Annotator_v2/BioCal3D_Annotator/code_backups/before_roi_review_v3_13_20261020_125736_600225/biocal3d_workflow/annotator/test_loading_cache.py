"""Cache freshness, shared-save isolation, reduced traversal and live feedback."""
import csv,json,os,tempfile,threading,time,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from local_cache import LocalReadCache
from mesh_io import atomic_npz,load_mesh,AnnotationStore,discover
from performance import MeshCache
from roi_review_store import read_reviewed,approve_displayed_surface,load_review_context
from annotation_features import AnnotatorFeatures,sample_frame

class LocalCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.source=self.root/'shared/geometry.npz';self.local=self.root/'local-cache'
        self.vertices=np.array([[-1.,-1,0],[1,-1,0],[1,1,0],[-1,1,0]])
        self.faces=np.array([[0,1,2],[0,2,3]])
        atomic_npz(self.source,vertices=self.vertices,faces=self.faces,rgb=np.tile([220,80,100],(4,1)),original_to_canonical=np.eye(4))
    def tearDown(self):self.tmp.cleanup()
    def test_copies_once_reuses_across_restart_and_keeps_logical_mesh_path(self):
        cache=LocalReadCache(self.local);before=self.source.read_bytes()
        cached=cache.materialize(self.source);self.assertNotEqual(cached,self.source);self.assertEqual(cached.read_bytes(),before)
        mesh=load_mesh(self.source,read_cache=cache);self.assertEqual(mesh.path,self.source)
        self.assertEqual(cache.copies,1);self.assertGreater(cache.hits,0)
        other=LocalReadCache(self.local);self.assertEqual(other.materialize(self.source),cached);self.assertEqual(other.copies,0)
        self.assertEqual(self.source.read_bytes(),before)
    def test_source_replacement_invalidates_cached_copy(self):
        cache=LocalReadCache(self.local);first=cache.materialize(self.source)
        atomic_npz(self.source,vertices=self.vertices+[0,0,.2],faces=self.faces,rgb=np.tile([220,80,100],(4,1)))
        second=cache.materialize(self.source);self.assertNotEqual(first,second)
        np.testing.assert_array_equal(load_mesh(self.source,read_cache=cache).vertices,self.vertices+[0,0,.2])
    def test_corrupt_local_copy_is_repaired_and_missing_source_cannot_use_stale_cache(self):
        cache=LocalReadCache(self.local);cached=cache.materialize(self.source);cached.write_bytes(b'corrupt')
        other=LocalReadCache(self.local);repaired=other.materialize(self.source)
        self.assertEqual(repaired.read_bytes(),self.source.read_bytes());self.assertEqual(other.copies,1)
        self.source.unlink()
        with self.assertRaises(FileNotFoundError):other.materialize(self.source)
    def test_unwritable_cache_falls_back_to_source(self):
        cache=LocalReadCache(self.local)
        with patch('local_cache.tempfile.mkstemp',side_effect=OSError('disk full')):
            self.assertEqual(cache.materialize(self.source),self.source)
        self.assertIn('disk full',cache.last_error)
    def test_no_labels_locks_or_json_are_copied(self):
        cache=LocalReadCache(self.local)
        for name in ['roi.ply.labels.npz','current.json','roi.ply.labels.npz.lock','roi.ply.labeled.ply']:
            path=self.root/'shared'/name;path.write_bytes(b'shared working file')
            self.assertEqual(cache.materialize(path),path)
        self.assertEqual(cache.copies,0);self.assertFalse(self.local.exists())
    def test_bounded_cache_evicts_old_inputs(self):
        cache=LocalReadCache(self.local,max_bytes=2500)
        paths=[]
        for n in range(4):
            source=self.root/f'shared/input{n}.ply';source.write_bytes(bytes([n])*800);paths.append(cache.materialize(source))
        size=sum(p.stat().st_size for p in self.local.rglob('*') if p.is_file())
        self.assertLessEqual(size,2500);self.assertTrue(paths[-1].is_file());self.assertFalse(paths[0].exists())
    def test_labels_and_review_checks_stay_in_shared_dataset(self):
        root=self.source.parent;store=AnnotationStore(root,root/'annotations','qa')
        store.mesh_cache=MeshCache();store.read_cache=LocalReadCache(self.local)
        approve_displayed_surface(store,read_reviewed(store,self.source))
        mesh=read_reviewed(store,self.source);store.save(mesh,np.array([1,0],np.int8),True,'shared label')
        self.assertTrue(store.path_for(self.source).is_relative_to(root/'annotations'))
        self.assertEqual(store.status(self.source),'done')
        other=AnnotationStore(root,root/'annotations','other');other.read_cache=store.read_cache
        np.testing.assert_array_equal(other.load(read_reviewed(other,self.source))[0],[-2,-2])
        # A new shared revision must not be bypassed by an old local input.
        from roi_review_store import review_dir
        state_file=review_dir(store,self.source)/'current.json';state=json.loads(state_file.read_text());state['version']='missing-new-version';state_file.write_text(json.dumps(state))
        with self.assertRaises(Exception):read_reviewed(store,self.source)
    def test_sample_frame_is_reused_and_invalidated_by_geometry_metadata(self):
        store=AnnotationStore(self.source.parent,self.source.parent/'annotations','qa');store.mesh_cache=MeshCache()
        mesh=read_reviewed(store,self.source);expected=sample_frame(mesh)
        with patch('annotation_features.np.linalg.svd',side_effect=AssertionError('repeated SVD')):
            actual=sample_frame(read_reviewed(store,self.source))
        for a,b in zip(expected[:3],actual[:3]):np.testing.assert_array_equal(a,b)

class FolderReadTests(unittest.TestCase):
    def test_discovery_does_not_descend_into_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sample=root/'samples/TRIOS5/G3/BCG3T1-1';sample.mkdir(parents=True);(sample/'roi.ply').write_text('fixture')
            excluded=root/'annotations/_roi_review/very/deep/history';excluded.mkdir(parents=True);(excluded/'old.ply').write_text('ignored')
            visited=[];walk=os.walk
            def tracked(*args,**kwargs):
                for row in walk(*args,**kwargs):visited.append(Path(row[0]));yield row
            with patch('mesh_io.os.walk',side_effect=tracked):result=discover(root)
            self.assertEqual(result,[sample/'roi.ply']);self.assertFalse(any(p.is_relative_to(root/'annotations') for p in visited))
    def test_manifest_metadata_avoids_per_sample_json_and_handles_review_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sample=root/'samples/TRIOS5/G3/BCG3T1-1';sample.mkdir(parents=True);path=sample/'roi.vtp';path.touch()
            with (root/'scan_manifest.csv').open('w',newline='') as out:
                writer=csv.DictWriter(out,fieldnames=['sample_dir','annotation_mesh','scanner','sample_id','stage']);writer.writeheader()
                writer.writerow(dict(sample_dir=str(sample.relative_to(root)),annotation_mesh=str((sample/'roi.ply').relative_to(root)),scanner='TRIOS5',sample_id='BCG3T1-1',stage='partial'))
            with patch('pathlib.Path.read_text',side_effect=AssertionError('per-sample JSON read')):
                info=AnnotatorFeatures.queue_metadata(None,root,[path])
            self.assertEqual(info[path]['stage'],'partial');self.assertEqual(info[path]['scanner'],'TRIOS5')

from PySide6 import QtCore,QtWidgets
from loading_ui import LoadingFeedback
class LoadingFeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    def setUp(self):self.owner=QtWidgets.QWidget()
    def tearDown(self):self.owner.close();self.owner.deleteLater();self.app.processEvents()
    def test_worker_leaves_timer_running_and_progress_reports_real_counts(self):
        ticks=[];timer=QtCore.QTimer();timer.setInterval(10);timer.timeout.connect(lambda:ticks.append(1));timer.start()
        main=threading.get_ident();previous_modal=self.app.activeModalWidget()
        with LoadingFeedback(self.owner,'Loading test sample') as feedback:
            def work():
                for n in range(5):feedback.report('Reading mesh',n+1,5);time.sleep(.03)
                return threading.get_ident()
            worker=feedback.run(work,'Reading specimen surface…')
            feedback.stage('Reading file',1024*1024,2*1024*1024)
            self.assertEqual(feedback.bar.value(),500);self.assertIn('Elapsed',feedback.elapsed.text())
            self.assertNotEqual(worker,main);self.assertGreater(len(ticks),3);self.assertEqual(self.owner._loading_depth,1)
        timer.stop();self.assertEqual(self.owner._loading_depth,0)
        self.assertIs(self.app.activeModalWidget(),previous_modal)
    def test_exception_stops_worker_and_releases_loading_state(self):
        with self.assertRaisesRegex(OSError,'unavailable'):
            with LoadingFeedback(self.owner,'Loading failed input') as feedback:
                def fail():raise OSError('share unavailable')
                feedback.run(fail,'Reading specimen surface…')
        self.assertEqual(self.owner._loading_depth,0)
        self.assertFalse(any(t.name=='BioCal3D-input-loader' for t in threading.enumerate()))


# Reuse the existing nonrendering application fixture for display preparation.
from test_quick_roi_review import QuickReviewTests as _Fixture
class PreparationTests(unittest.TestCase):
    setUpClass=classmethod(lambda cls:setattr(cls,'app',QtWidgets.QApplication.instance() or QtWidgets.QApplication([])))
    setUp=_Fixture.setUp
    tearDown=_Fixture.tearDown
    def test_normals_are_lazy_and_initial_view_render_is_batched(self):
        w=self.owner;w.shading_picker.setCurrentIndex(0)
        with patch('biocal3d_annotator.shaded_display',wraps=__import__('biocal3d_annotator').shaded_display) as normals:
            render_window=w.view.GetRenderWindow()
            with patch.object(render_window,'Render',wraps=render_window.Render) as render:
                w.change_sample(1)
            self.assertEqual(normals.call_count,0);self.assertEqual(render.call_count,1)
            w.shading_picker.setCurrentIndex(1);self.assertEqual(normals.call_count,1)
            self.assertIsNotNone(w.view.render_data.GetPointData().GetNormals())
        self.assertFalse(self.errors,self.errors)
    def test_second_open_reads_current_labels_while_keeping_cached_logical_paths(self):
        w=self.owner;w.edit(np.array([0]),1);w.save();source=w.mesh.path;annotation=w.store.path_for(source)
        self.assertTrue(annotation.is_relative_to(w.root/'annotations'))
        self.assertFalse(source.is_relative_to(w.read_cache.root))
        w.change_sample(1);w.change_sample(0)
        self.assertEqual(w.mesh.path,source);np.testing.assert_array_equal(w.labels,[1,-2]);self.assertFalse(self.errors,self.errors)
del _Fixture

if __name__=='__main__':unittest.main()
