"""Geometry, identity and label transfer regression tests."""
import sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from roi_geometry import Extraction,select_surface
from mesh_io import Mesh,polydata,atomic_npz,load_mesh,AnnotationStore,export_ply,mesh_appearance
from roi_review_store import load_review_context,save_review,read_reviewed,current_version

class SurfaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.v=np.array([[-1.,-1,0],[1,-1,0],[1,1,0],[-1,1,0],[1,-1,-1],[1,1,-1],
                         [2,-1,-.8],[3,-1,-.8],[2,0,-.8]])
        self.f=np.array([[0,1,2],[0,2,3],[1,4,5],[1,5,2],[6,7,8]])
        self.source=self.root/'sample'/'original.ply';self.source.parent.mkdir()
        m=Mesh(self.root/'input.ply',self.v,self.f,polydata(self.v,self.f,np.tile([220,80,100],(9,1))),np.ones(5,bool))
        export_ply(m,None,self.source);self.mesh=load_mesh(self.source)
        self.store=AnnotationStore(self.root,self.root/'annotations','dentist1');self.settings=Extraction(radius=4,below=1.5,above=1.5)
    def tearDown(self):self.tmp.cleanup()
    def test_sidewall_and_holder_removed(self):
        mask,metrics=select_surface(self.v,self.f,self.settings)
        np.testing.assert_array_equal(mask,[True,True,False,False,False])
        self.assertIn('multiple_surface_components',metrics['flags']);self.assertIn('steep_faces_removed',metrics['flags'])
    def test_touching_vertex_does_not_connect_sheet(self):
        v=np.r_[self.v,[[0,0,-1],[.2,0,-1]]];f=np.r_[self.f[:2],[[0,9,10]]]
        mask,_=select_surface(v,f,Extraction(radius=4,max_angle=90))
        np.testing.assert_array_equal(mask,[True,True,False])
    def test_real_relief_preserved(self):
        v=self.v[:4].copy();v[:,2]=[0,.2,.3,.1];before=v.copy()
        mask,_=select_surface(v,self.f[:2],self.settings)
        self.assertTrue(mask.all());np.testing.assert_array_equal(v,before)
    def test_automatic_selection_ignores_uv_vertex_splits_and_is_repeatable(self):
        # Each triangle has separate corners, exactly as a UV-seamed scanner export.
        v=self.v[self.f[:2].ravel()];f=np.arange(6).reshape(-1,3)
        before_v=v.copy();before_f=f.copy()
        for _ in range(3):
            mask,metrics=select_surface(v,f,self.settings)
            np.testing.assert_array_equal(mask,[True,True])
            self.assertEqual(metrics['components'],1)
        np.testing.assert_array_equal(v,before_v);np.testing.assert_array_equal(f,before_f)

    def test_close_separate_surfaces_are_not_welded(self):
        v=np.array([[0.,0,0],[1,0,0],[0,1,0],[1,0,1e-7],[0,1,1e-7],[1,1,1e-7]])
        f=np.array([[0,1,2],[3,5,4]])
        mask,metrics=select_surface(v,f,Extraction(radius=3))
        self.assertEqual(metrics['components'],2);self.assertEqual(mask.sum(),1)

    def detail_fixture(self,height=.4):
        x,y=np.meshgrid(np.linspace(-1,1,5),np.linspace(-1,1,5))
        v=np.c_[x.ravel(),y.ravel(),np.zeros(25)];v[12,2]=height
        f=[]
        for j in range(4):
            for i in range(4):
                a=j*5+i;f.extend([[a,a+1,a+5],[a+1,a+6,a+5]])
        # Exposed vertical sidewall shares the boundary of the real top sheet.
        v=np.r_[v,[[1,-1,-.4],[1,-.5,-.4]]]
        return v,np.r_[np.array(f),[[4,25,26],[4,26,9]]]

    def test_enclosed_steep_detail_kept_but_sidewalls_removed_with_uv_splits(self):
        v,f=self.detail_fixture();v=v[f.ravel()];f=np.arange(len(v)).reshape(-1,3)
        before=v.copy();settings=Extraction(radius=2,max_angle=25)
        mask,metrics=select_surface(v,f,settings)
        np.testing.assert_array_equal(mask,np.r_[np.ones(32,bool),False,False])
        self.assertGreater(metrics['enclosed_detail_faces'],0)
        self.assertEqual(metrics['steep_excluded_faces'],2)
        np.testing.assert_array_equal(v,before)
        again,_=select_surface(v,f,settings);np.testing.assert_array_equal(again,mask)

    def test_enclosed_detail_cannot_bypass_height_or_circle(self):
        v,f=self.detail_fixture(height=.6)
        mask,metrics=select_surface(v,f,Extraction(radius=2,max_angle=25))
        self.assertFalse(mask[np.any(f==12,axis=1)].any())
        self.assertGreater(metrics['height_excluded_faces'],0)
        v,f=self.detail_fixture()
        settings=Extraction(cx=-.5,radius=.8,max_angle=25)
        mask,_=select_surface(v,f,settings)
        self.assertTrue(np.all(np.linalg.norm(v[f[mask]][:,:,:2]-[-.5,0],axis=2)<=.8+1e-10))

    def test_migration_archive_and_stale_writer(self):
        self.store.save(self.mesh,np.array([1,0,-1,-2,0],np.int8),False,'old labels')
        base,mask,s,_,_=load_review_context(self.store,self.mesh);mask[:]=[1,1,0,0,0]
        save_review(self.store,self.source,base,mask,s,'approved','remove holder','')
        reviewed=read_reviewed(self.store,self.source);labels,done,notes=self.store.load(reviewed)
        np.testing.assert_array_equal(labels,[1,0]);self.assertFalse(done)
        self.store.save(reviewed,labels,False,notes)
        self.assertTrue(list(self.store.path_for(self.source).parent.glob('.geometry_history/*')))
        with self.assertRaises(ValueError):self.store.save(self.mesh,np.zeros(5,np.int8),False,'stale')
        with self.assertRaises(RuntimeError):save_review(self.store,self.source,base,mask,s,'approved','','')
        version=current_version(self.store,self.source);mask[4]=True
        save_review(self.store,self.source,base,mask,s,'pending','restore',version)
        revised=read_reviewed(self.store,self.source);lab,done,_=self.store.load(revised)
        np.testing.assert_array_equal(lab,[1,0,-2]);self.assertFalse(done)
    def test_same_geometry_review_resets_completion(self):
        self.store.save(self.mesh,np.zeros(5,np.int8),True,'done')
        base,mask,s,_,_=load_review_context(self.store,self.mesh)
        save_review(self.store,self.source,base,mask,s,'approved','','')
        labels,done,_=self.store.load(read_reviewed(self.store,self.source))
        self.assertFalse(done);self.assertTrue((labels==0).all())
    def test_full_raw_context_and_restoration(self):
        raw=self.root/'raw.ply';export_ply(self.mesh,None,raw)
        v=self.v[:4];f=self.f[:2];p=self.root/'prepared'/'roi.ply';p.parent.mkdir()
        m=Mesh(self.source,v,f,polydata(v,f,np.tile([220,80,100],(4,1))),np.ones(2,bool))
        export_ply(m,None,p,face_ids=np.array([0,1]))
        atomic_npz(p.with_name('geometry.npz'),vertices=v,faces=f,raw_triangle_id=np.array([0,1]),raw_vertex_id=np.arange(4),
                   original_to_canonical=np.eye(4),source_ply=np.array(str(raw)),roi_radius=np.array(4.))
        m=load_mesh(p);base,mask,s,_,_=load_review_context(self.store,m)
        self.assertEqual(len(base.faces),5);np.testing.assert_array_equal(mask,[1,1,0,0,0])
        self.store.save(m,np.array([1,0],np.int8),True,'original');mask[4]=True
        save_review(self.store,p,base,mask,s,'approved','restore','',str(raw))
        reviewed=read_reviewed(self.store,p);lab,done,_=self.store.load(reviewed)
        np.testing.assert_array_equal(lab,[1,0,-2]);self.assertFalse(done)
        np.testing.assert_array_equal(reviewed.source_face_ids,[0,1,4])
    def test_uv_seams_survive(self):
        from PIL import Image
        uv=np.array([[0,0],[1,0],[1,1],[.3,0],[1,1],[0,1]],float);v=self.v[:4];f=self.f[:2]
        image=self.root/'texture.png';Image.fromarray(np.zeros((4,4,3),np.uint8)).save(image)
        m=Mesh(self.source,v,f,polydata(v[f.ravel()],np.arange(6).reshape(-1,3),None,uv),np.ones(2,bool),image)
        other=self.root/'texture.ply';export_ply(m,None,other);m=load_mesh(other)
        base,mask,s,_,_=load_review_context(self.store,m);save_review(self.store,other,base,mask,s,'approved','','')
        result=read_reviewed(self.store,other)
        np.testing.assert_allclose(mesh_appearance(result)[1],uv.reshape(2,6));self.assertTrue(result.texture.is_file())
    def test_uploaded_roi(self):
        example=Path(__file__).resolve().parents[3]/'examples'/'roi.vtp.labeled.ply'
        if not example.exists():self.skipTest('Local experimental example not bundled')
        m=load_mesh(example);self.assertGreater(len(m.faces),100)

if __name__=='__main__':unittest.main()
