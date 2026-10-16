"""Actual prepared metadata -> VTP approvals -> rebuilt canonical datasets."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pyvista as pv
from PIL import Image
import biocal3d_prepare_all as prep
from rebuild_reviewed_snapshot import rebuild,normalize_export_row
from mesh_io import AnnotationStore,load_mesh,Mesh,polydata,export_ply
from roi_review_store import load_review_context,save_review,read_reviewed,approve_displayed_surface


class PreparedReviewPipelineTests(unittest.TestCase):
    def test_real_preparation_schema_t0_t1_t2_rgb_texture_and_repeat_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'raw';seed=root/'seed';prepared=root/'prepared';verified=root/'verified'
            vertices=np.array([[-1.,-1,0],[1,-1,0],[-1,1,0],[1,1,0]])
            faces=np.array([[0,1,2],[1,3,2]])
            for scanner in ('TRIOS5','iTERO'):
                for time in range(3):
                    source=raw/scanner/'G3'/f'BCG3T{time}-1'/'source.ply';source.parent.mkdir(parents=True)
                    surface=pv.PolyData(vertices,np.c_[np.full(2,3),faces])
                    surface.point_data['RawVertexId']=np.arange(4);surface.cell_data['RawTriangleId']=np.arange(2)
                    rgb=np.tile([210,40,110],(4,1)).astype(np.uint8);surface.point_data['RGB']=rgb;surface.save(source)
                    row=prep.identify(source,raw)
                    row.update(source=str(source),status='OK',registration_method='SYNTHETIC',holder_fitness=1.,holder_rmse=.1)
                    uv=np.array([[0.,0],[1,0],[0,1],[1,1]])
                    texture=np.zeros((4,4,3),np.float32);texture[:]=[.8,.2,.4]
                    image=source.with_suffix('.png');Image.fromarray(np.uint8(texture*255)).save(image)
                    textured=scanner=='iTERO'
                    if textured:surface.active_texture_coordinates=uv
                    colors=dict(rgb=None if textured else rgb/255.,uv=uv if textured else None,
                                texture=texture if textured else None,color_source='texture_uv' if textured else 'vertex_rgb',
                                texture_path=str(image) if textured else '')
                    # Write scanner-style RGB/UV fields explicitly, so full-source
                    # correction has the same verified appearance as preparation.
                    raw_mesh=Mesh(root/'input',vertices,faces,polydata(vertices,faces,rgb,uv if textured else None),
                                  np.ones(2,bool),image if textured else None)
                    export_ply(raw_mesh,None,source,face_ids=np.arange(2))
                    prep.export_sample(row,surface,colors,np.eye(4),1.5,seed/scanner/'G3'/row['sample_id'],32,4.,True)
            args=prep.parser().parse_args(['prepare','--data-root',str(raw),'--processed-root',str(seed),
                                          '--output',str(prepared),'--size','32','--fov-mm','4','--no-context-qc'])
            original_write=prep.write_json
            with patch.object(prep,'write_json',side_effect=lambda path,value:None if Path(path).name=='last_prepared.json' else original_write(path,value)):
                self.assertEqual(prep.prepare(args),0)
            scans=json.loads((prepared/'dataset_index.json').read_text())['scans']
            self.assertEqual(len(scans),6)
            store=AnnotationStore(prepared,prepared/'annotations','johan')
            for i,row in enumerate(scans):
                source=prepared/row['sample_dir']/'roi.vtp';mesh=load_mesh(source)
                if i==0:
                    base,mask,settings,_,rawpath=load_review_context(store,mesh);mask[1]=False
                    save_review(store,source,base,mask,settings,'approved','Manual correction','',rawpath)
                else:approve_displayed_surface(store,mesh)
                mesh=read_reviewed(store,source);store.save(mesh,np.ones(len(mesh.faces),np.int8),True,'Labels preserved')
            rebuild(prepared,verified)
            output=json.loads((verified/'dataset_index.json').read_text())['scans']
            self.assertEqual(len(output),6)
            for i,row in enumerate(output):
                self.assertIn(row['timepoint'],('T0','T1','T2'))
                self.assertEqual(row['timepoint'],f'T{row["time_idx"]}')
                self.assertIsInstance(row['time_idx'],int);self.assertIsInstance(row['group'],int)
                self.assertIsInstance(row['sample_number'],int)
                new_store=AnnotationStore(verified,verified/'annotations','johan')
                mesh=read_reviewed(new_store,verified/row['annotation_mesh'])
                labels,done,_=new_store.load(mesh)
                self.assertEqual(len(mesh.faces),1 if i==0 else 2)
                self.assertTrue(done);self.assertTrue((labels==1).all());self.assertEqual(mesh.review_status,'approved')
            rebuild(verified,root/'verified_again')
            self.assertEqual(json.loads((root/'verified_again/geometry_rebuild.json').read_text())['status'],'complete')

    def test_legacy_numeric_timepoints_and_conflicting_fields(self):
        row=normalize_export_row(dict(group='3',timepoint='0',sample_number='1'))
        self.assertEqual((row['group'],row['timepoint'],row['time_idx'],row['sample_number']),(3,'T0',0,1))
        with self.assertRaisesRegex(ValueError,'conflicts'):
            normalize_export_row(dict(scan_id='example',group='3',timepoint='T0',time_idx='1'))
        with self.assertRaisesRegex(ValueError,'invalid timepoint'):
            normalize_export_row(dict(group='3',timepoint='baseline'))


if __name__=='__main__':unittest.main()
