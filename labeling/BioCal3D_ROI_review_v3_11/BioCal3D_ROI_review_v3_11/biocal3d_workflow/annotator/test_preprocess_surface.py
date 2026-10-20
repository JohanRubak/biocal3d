"""Exercise the actual preprocessing entry point without registration hardware/data."""
import sys,types,unittest
from pathlib import Path
import numpy as np
import pyvista as pv
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
# Open3D is only used by registration routines, not by extract_specimen.
try:import open3d
except ImportError:sys.modules['open3d']=types.ModuleType('open3d')
import biocal3d_roi_core as core

class PipelineTests(unittest.TestCase):
    def test_actual_extractor_preserves_raw_ids_and_filters_sidewall(self):
        x,y=np.meshgrid(np.linspace(-2,2,11),np.linspace(-2,2,11));v=np.c_[x.ravel(),y.ravel(),np.zeros(121)]
        f=[]
        for j in range(10):
            for i in range(10):
                a=j*11+i;f.extend([(a,a+1,a+11),(a+1,a+12,a+11)])
        v=np.r_[v,[[2,-2,-1],[2,-1.6,-1]]];f=np.r_[np.asarray(f),[[10,121,122],[10,122,21]]]
        mesh=pv.PolyData(v,np.c_[np.full(len(f),3),f]);mesh.point_data['RGB']=np.tile([200,50,100],(len(v),1))
        mesh.point_data['RawVertexId']=np.arange(len(v));mesh.cell_data['RawTriangleId']=np.arange(len(f))
        reference=dict(center=np.zeros(3),normal=np.array([0,0,1.]),u_axis=np.array([1.,0,0]),v_axis=np.array([0,1.,0]),radius=3.,z_offset=0.)
        result,offset,radius,transform=core.extract_specimen(mesh,reference)
        self.assertEqual(result.n_cells,200);self.assertEqual(offset,0.)
        np.testing.assert_array_equal(result.cell_data['RawTriangleId'],np.arange(200))
        self.assertTrue(np.all(result.points[:,2]==0));self.assertIn('ROIExtractionJSON',result.field_data)

if __name__=='__main__':unittest.main()
