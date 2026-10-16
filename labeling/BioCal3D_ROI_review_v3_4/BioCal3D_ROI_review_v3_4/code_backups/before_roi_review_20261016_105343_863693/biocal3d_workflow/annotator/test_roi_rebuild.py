import json,sys,tempfile,unittest,csv
from pathlib import Path
import numpy as np
import pyvista as pv
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from biocal3d_prepare_all import export_sample,write_csv
from rebuild_reviewed_snapshot import rebuild
from mesh_io import Mesh,polydata,export_ply,load_mesh,AnnotationStore
from roi_review_store import load_review_context,save_review,read_reviewed,current_version
from audit_roi_geometry import audit

class RebuildTests(unittest.TestCase):
    def test_full_rebuild_keeps_splits_labels_ids_and_regenerates_maps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'snapshot';root.mkdir();out=Path(tmp)/'corrected'
            v=np.array([[-1.,-1,0],[1,-1,0],[1,1,0],[-1,1,0],[1,-1,-1],[1,1,-1]])
            f=np.array([[0,1,2],[0,2,3],[1,4,5],[1,5,2]])
            raw=Path(tmp)/'raw.ply';colors=np.tile([220,80,100],(6,1)).astype(np.uint8)
            m=Mesh(Path(tmp)/'input',v,f,polydata(v,f,colors),np.ones(4,bool));export_ply(m,None,raw)
            p=pv.PolyData(v,np.c_[np.full(4,3),f]);p.point_data['RawVertexId']=np.arange(6);p.cell_data['RawTriangleId']=np.arange(4)
            row=dict(source=str(raw),scanner='TRIOS5',group=5,timepoint=1,sample_id='BCG5T1-1',scan_id='TRIOS5_BCG5T1-1',
                     physical_id='G5_S1',split='test',stage='partial',registration_method='REFERENCE',holder_fitness=1.,holder_rmse=0.)
            folder=root/'samples'/'case';folder.mkdir(parents=True)
            info=export_sample(row,p,dict(rgb=colors/255.,uv=None,texture=None,color_source='vertex_rgb',texture_path=''),np.eye(4),2.,folder,32,6.,True)
            entry={**info,'sample_dir':'samples/case','annotation_mesh':'samples/case/roi.ply'}
            write_csv(root/'scan_manifest.csv',[entry]);write_csv(root/'annotation_inventory.csv',[entry])
            (root/'specimen_splits.csv').write_text('physical_id,split\nG5_S1,test\n')
            store=AnnotationStore(root,root/'annotations','dentist1');source=folder/'roi.ply';original=load_mesh(source)
            base,mask,s,_,_=load_review_context(store,original);mask[:]=[1,1,0,0]
            save_review(store,source,base,mask,s,'approved','remove walls','',str(raw))
            reviewed=read_reviewed(store,source);store.save(reviewed,np.array([1,0],np.int8),True,'reviewed')
            rebuild(root,out)
            new_source=out/'samples/case/roi.ply';new_store=AnnotationStore(out,out/'annotations','dentist1')
            new_mesh=read_reviewed(new_store,new_source);labels,done,_=new_store.load(new_mesh)
            np.testing.assert_array_equal(labels,[1,0]);self.assertTrue(done);self.assertEqual(new_mesh.review_status,'approved')
            np.testing.assert_array_equal(new_mesh.source_face_ids,[0,1]);self.assertEqual(len(new_mesh.faces),2)
            self.assertEqual((out/'specimen_splits.csv').read_bytes(),(root/'specimen_splits.csv').read_bytes())
            with np.load(out/'samples/case/maps.npz',allow_pickle=False) as maps:
                self.assertEqual(len(maps['faces']),2);self.assertTrue(np.all(maps['triangle_id'][maps['valid']]<2))
            self.assertTrue(json.loads((out/'geometry_rebuild.json').read_text())['status']=='complete')
            report=audit(root,Path(tmp)/'audit',limit=1);self.assertEqual(len(report),1);self.assertFalse(report[0]['error'])
            # Original geometry is still the four-face mesh.
            self.assertEqual(len(load_mesh(source).faces),4)

    def test_textured_rebuild_does_not_duplicate_shared_vertices(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'snapshot';root.mkdir();out=Path(tmp)/'corrected'
            v=np.array([[-1.,-1,0],[1,-1,0],[1,1,0],[-1,1,0]]);f=np.array([[0,1,2],[0,2,3]])
            uv=np.array([[0,0],[1,0],[1,1],[0,1]],float)
            tex=np.zeros((8,8,3),np.uint8);tex[:4]=[230,40,80];tex[4:]=[50,120,210]
            texture=Path(tmp)/'texture.png';Image.fromarray(tex).save(texture)
            raw=Path(tmp)/'raw.ply';m=Mesh(Path(tmp)/'input',v,f,polydata(v,f,None,uv),np.ones(2,bool),texture)
            export_ply(m,None,raw)
            surface=pv.PolyData(v,np.c_[np.full(2,3),f]);surface.point_data['RawVertexId']=np.arange(4);surface.cell_data['RawTriangleId']=np.arange(2)
            row=dict(source=str(raw),scanner='LABscanner',group=5,timepoint=1,sample_id='BCG5T1-1',scan_id='LAB_BCG5T1-1',
                     physical_id='G5_S1',split='val',stage='partial',registration_method='REFERENCE',holder_fitness=1.,holder_rmse=0.)
            folder=root/'samples/case';folder.mkdir(parents=True)
            info=export_sample(row,surface,dict(rgb=None,uv=uv,texture=tex/255.,color_source='texture_uv',texture_path=str(texture)),np.eye(4),2.,folder,32,6.,True)
            entry={**info,'sample_dir':'samples/case','annotation_mesh':'samples/case/roi.ply'}
            write_csv(root/'scan_manifest.csv',[entry]);write_csv(root/'annotation_inventory.csv',[entry])
            store=AnnotationStore(root,root/'annotations','qa');source=folder/'roi.ply'
            base,mask,settings,_,rawpath=load_review_context(store,load_mesh(source))
            save_review(store,source,base,mask,settings,'approved','','',rawpath)
            rebuild(root,out)
            with np.load(out/'samples/case/geometry.npz',allow_pickle=False) as z:
                self.assertEqual(len(z['vertices']),4);np.testing.assert_allclose(z['uv'],uv)
            np.testing.assert_array_equal(np.array(Image.open(folder/'rgb.png')),np.array(Image.open(out/'samples/case/rgb.png')))

if __name__=='__main__':unittest.main()
