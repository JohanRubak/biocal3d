"""Prepared-format identity: reuse old approvals/corrections without renaming."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from mesh_io import Mesh,polydata,export_ply,load_mesh,atomic_npz,AnnotationStore,discover
from roi_review_store import (load_review_context,save_review,read_reviewed,read_state,
                              resolve_review_source,manifest_reviews,approve_displayed_surface)


class ReviewAliasTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.folder=self.root/'samples/iTERO/G1/BCG1T0-1';self.folder.mkdir(parents=True)
        self.v=np.array([[-1.1234567,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],float)
        self.f=np.array([[0,1,2],[0,2,3]])
        self.ids=np.array([10,11]);self.colors=np.tile([220,80,100],(4,1))
        self.ply=self.folder/'roi.ply';self.vtp=self.folder/'roi.vtp'
        self.store=AnnotationStore(self.root,self.root/'annotations','qa')
        self.write_inputs()
        import csv
        with (self.root/'scan_manifest.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['scan_id','sample_dir','annotation_mesh']);writer.writeheader()
            writer.writerow(dict(scan_id='iTERO__BCG1T0-1',sample_dir=str(self.folder.relative_to(self.root)),annotation_mesh=str(self.ply.relative_to(self.root))))

    def write_inputs(self):
        p=polydata(self.v,self.f,self.colors)
        a=numpy_to_vtk(self.ids,deep=True);a.SetName('RawTriangleId');p.GetCellData().AddArray(a)
        writer=vtk.vtkXMLPolyDataWriter();writer.SetFileName(str(self.vtp));writer.SetInputData(p);writer.Write()
        m=Mesh(self.root/'input',self.v,self.f,p,np.ones(2,bool))
        export_ply(m,None,self.ply,face_ids=self.ids)
        atomic_npz(self.folder/'geometry.npz',vertices=self.v,faces=self.f,rgb=self.colors,raw_triangle_id=self.ids)

    def approve_vtp(self,trim=False):
        base,mask,settings,_,raw=load_review_context(self.store,load_mesh(self.vtp))
        if trim:mask[1]=False
        return save_review(self.store,self.vtp,base,mask,settings,'approved','Existing review','','')

    def tearDown(self):self.tmp.cleanup()

    def test_existing_vtp_correction_and_labels_found_from_ply(self):
        self.approve_vtp(trim=True)
        original_review=self.store.output/'_roi_review'/self.vtp.relative_to(self.root)/'current.json'
        before=original_review.read_bytes()
        mesh=read_reviewed(self.store,self.vtp);self.store.save(mesh,np.array([1],np.int8),True,'Plaque note')
        resolved=resolve_review_source(self.store,self.ply);self.assertEqual(resolved,self.vtp)
        mesh=read_reviewed(self.store,self.ply);self.assertEqual(mesh.path,self.vtp)
        self.assertEqual(mesh.review_status,'approved');self.assertEqual(len(mesh.faces),1)
        np.testing.assert_array_equal(mesh.source_face_ids,[10])
        labels,done,notes=self.store.load(mesh);np.testing.assert_array_equal(labels,[1]);self.assertTrue(done)
        self.assertEqual(notes,'Plaque note');self.assertEqual(original_review.read_bytes(),before)
        checks=manifest_reviews(self.store);self.assertEqual(checks[0]['status'],'approved');self.assertEqual(checks[0]['source'],self.vtp)
        self.assertFalse((original_review.parent.parent/'roi.ply').exists())

    def test_different_geometry_or_source_ids_cannot_reuse_approval(self):
        self.approve_vtp();self.v[0,0]-=.1
        m=Mesh(self.root/'input',self.v,self.f,polydata(self.v,self.f,self.colors),np.ones(2,bool))
        export_ply(m,None,self.ply,face_ids=self.ids)
        with self.assertRaisesRegex(ValueError,'different input geometry'):resolve_review_source(self.store,self.ply)
        self.v[0,0]+=.1
        m=Mesh(self.root/'input',self.v,self.f,polydata(self.v,self.f,self.colors),np.ones(2,bool))
        export_ply(m,None,self.ply,face_ids=self.ids+1)
        with self.assertRaisesRegex(ValueError,'different source-face IDs'):resolve_review_source(self.store,self.ply)

    def test_multiple_histories_are_reported_without_overwriting_either(self):
        self.approve_vtp()
        # A second app/version could have saved independently under roi.ply.
        base=load_mesh(self.ply);base.source_vertex_ids=np.arange(len(base.vertices));base.source_domain='test'
        from roi_geometry import Extraction
        save_review(self.store,self.ply,base,base.active,Extraction(radius=3),'approved','','')
        with self.assertRaisesRegex(ValueError,'Multiple ROI review histories'):resolve_review_source(self.store,self.ply)
        self.assertEqual(read_state(self.store,self.ply)['status'],'approved')
        self.assertEqual(read_state(self.store,self.vtp)['status'],'approved')

    def test_discovery_prefers_manifest_format_for_new_reviews(self):
        self.assertEqual(discover(self.root),[self.ply])
        self.assertEqual(resolve_review_source(self.store,self.ply),self.ply)
        checks=manifest_reviews(self.store);self.assertEqual(checks[0]['status'],'pending')

    def test_custom_annotation_root_is_respected(self):
        default_store=self.store
        self.store=AnnotationStore(self.root,self.root/'shared_reviews','qa');self.approve_vtp()
        self.assertEqual(resolve_review_source(self.store,self.ply),self.vtp)
        self.assertEqual(manifest_reviews(default_store)[0]['status'],'pending')


if __name__=='__main__':unittest.main()
