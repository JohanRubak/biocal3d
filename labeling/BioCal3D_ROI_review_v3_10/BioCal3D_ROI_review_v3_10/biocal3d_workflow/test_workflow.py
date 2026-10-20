"""Synthetic integration checks; does not replace real scanner/registration QC."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import pyvista as pv
from biocal3d_prepare_all import export_sample,prepare,parser,identify,launch_labeler
from annotator.mesh_io import load_mesh,manifest_queue

class WorkflowTests(unittest.TestCase):
    def test_prepared_pipeline_ply_texture_queue_and_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'raw';old=root/'old';out=root/'new'
            vertices=np.array([[-1.,-1,0],[1,-1,0],[-1,1,0],[1,1,0]])
            faces=np.array([[0,1,2],[1,3,2]])
            for scanner,number,texture_mode,fitness in [('TRIOS3',1,False,.8),('LABscanner',1,True,.8),('TRIOS5',2,False,.1)]:
                source=raw/scanner/'G3'/f'BCG3T1-{number}'/'source.ply'
                source.parent.mkdir(parents=True)
                mesh=pv.PolyData(vertices,np.column_stack((np.full(2,3),faces)).ravel())
                mesh.point_data['RawVertexId']=np.arange(4)
                mesh.cell_data['RawTriangleId']=np.arange(2)
                mesh.point_data['RGB']=np.tile([210,40,110],(4,1)).astype(np.uint8)
                mesh.save(source)
                row=identify(source,raw)
                row.update(source=str(source),status='OK',registration_method='SYNTHETIC',holder_fitness=fitness,holder_rmse=.1)
                uv=np.array([[0.,0],[1,0],[0,1],[1,1]])
                texture=np.zeros((4,4,3),np.float32);texture[:,:,0]=np.linspace(0,1,4)[None,:];texture[:,:,1]=np.linspace(0,1,4)[:,None]
                image=source.with_suffix('.png');Image.fromarray(np.uint8(texture*255)).save(image)
                if texture_mode:mesh.active_texture_coordinates=uv
                colors=dict(rgb=None if texture_mode else np.tile([210/255,40/255,110/255],(4,1)),
                            uv=uv if texture_mode else None,texture=texture if texture_mode else None,
                            color_source='texture_uv' if texture_mode else 'vertex_rgb',texture_path=str(image) if texture_mode else '')
                folder=old/scanner/'G3'/row['sample_id']
                export_sample(row,mesh,colors,np.eye(4),1.5,folder,32,4,True)
                restored=load_mesh(folder/'roi.ply')
                np.testing.assert_array_equal(restored.vertices,vertices)
                np.testing.assert_array_equal(restored.faces,faces)
                if texture_mode:self.assertEqual(restored.texture.read_bytes(),image.read_bytes())
            args=parser().parse_args(['prepare','--data-root',str(raw),'--processed-root',str(old),'--output',str(out),'--size','32','--fov-mm','4','--no-context-qc'])
            # Avoid persisting the synthetic snapshot in the real script's last-run pointer.
            import biocal3d_prepare_all as module
            original_write=module.write_json
            def write(path,value):
                if Path(path).name=='last_prepared.json':return
                return original_write(path,value)
            with patch.object(module,'write_json',side_effect=write):
                self.assertEqual(prepare(args),0)
            index=json.loads((out/'dataset_index.json').read_text())
            self.assertEqual(len(index['scans']),3)
            queued=manifest_queue(out,out/'annotation_inventory.csv')
            self.assertEqual(len(queued),2)
            self.assertTrue(all(p.name=='roi.ply' for p in queued))
            self.assertFalse((out/'x_anylabeling').exists())
            self.assertTrue(all((out/r['sample_dir']/'maps.npz').is_file() for r in index['scans']))
            lab=parser().parse_args(['label','--output',str(out),'--annotator','rater2'])
            with patch.object(module.subprocess,'call',return_value=0) as run:
                self.assertEqual(launch_labeler(lab),0)
                command=run.call_args.args[0]
                self.assertIn('--manifest',command);self.assertIn('rater2',command)
            # Frozen-input verification rejects a changed PLY before opening the queue.
            source=queued[0];original=source.read_bytes();source.write_bytes(original+b'changed')
            with self.assertRaises(ValueError):manifest_queue(out,out/'annotation_inventory.csv')
            source.write_bytes(original)
            # Inventory paths cannot escape the dataset.
            invalid=out/'invalid.csv';invalid.write_text('annotation_mesh\n../elsewhere.ply\n')
            with self.assertRaises(ValueError):manifest_queue(out,invalid)

if __name__=='__main__':unittest.main()
