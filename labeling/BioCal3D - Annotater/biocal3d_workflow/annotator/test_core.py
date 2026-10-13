"""Run: python -m unittest -v test_core.py"""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from mesh_io import (AnnotationStore, area_summary, atomic_npz, discover, export_vtp,
                     export_ply, load_mesh, create_demo, prepare_roi_ply, shaded_display)

class CoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.v=np.array([[0.,0,0],[1,0,0],[0,1,0],[1,1,0]])
        self.f=np.array([[0,1,2],[1,3,2]])
        self.source=self.root/'sample'/'geometry.npz'
        atomic_npz(self.source,vertices=self.v,faces=self.f,rgb=np.tile([1.,.2,.5],(4,1)))
    def tearDown(self):
        self.tmp.cleanup()
    def test_roundtrip_and_changed_mesh_rejected(self):
        mesh=load_mesh(self.source); store=AnnotationStore(self.root,self.root/'annotations','rater1')
        labels,done,notes=store.load(mesh)
        np.testing.assert_array_equal(labels,[-2,-2]); self.assertFalse(done)
        with self.assertRaises(ValueError):
            store.save(mesh,labels,True,'')
        labels[:]=[1,0]; store.save(mesh,labels,True,'edge reviewed',{'parallel_scale':4})
        reopened,done,notes=store.load(mesh)
        np.testing.assert_array_equal(reopened,labels); self.assertTrue(done); self.assertEqual(notes,'edge reviewed')
        self.assertEqual(store.status(self.source),'done')
        other=load_mesh(self.source); other.faces=other.faces[::-1]
        with self.assertRaises(ValueError):
            store.load(other)
        np.testing.assert_array_equal(np.load(self.source)['faces'],self.f)
    def test_independent_raters(self):
        mesh=load_mesh(self.source)
        a=AnnotationStore(self.root,self.root/'annotations','rater1')
        b=AnnotationStore(self.root,self.root/'annotations','rater2')
        a.save(mesh,np.array([1,0],np.int8),True,'')
        self.assertEqual(b.status(self.source),'new')
        np.testing.assert_array_equal(b.load(mesh)[0],[-2,-2])
    def test_roi_and_area(self):
        atomic_npz(self.source,vertices=self.v,faces=self.f,face_roi_mask=[True,False])
        mesh=load_mesh(self.source); store=AnnotationStore(self.root,self.root/'annotations','rater')
        labels=store.load(mesh)[0]; np.testing.assert_array_equal(labels,[-2,-1])
        labels[0]=1; store.save(mesh,labels,True,'')
        self.assertEqual(area_summary(mesh,labels)['plaque'],.5)
        self.assertEqual(area_summary(mesh,labels)['uncertain'],0)
    def test_ply_wedge_uv_face_order(self):
        path=self.root/'sample.ply'
        path.write_text('''ply
format ascii 1.0
comment TextureFile sample.jpg
element vertex 4
property float x
property float y
property float z
element face 2
property list uchar int vertex_indices
property list uchar float texcoord
end_header
0 0 0
1 0 0
0 1 0
1 1 0
3 0 1 2 6 0 0 1 0 0 1
3 1 3 2 6 1 0 1 1 0 1
''')
        (self.root/'sample.jpg').write_bytes(b'texture filename test')
        mesh=load_mesh(path)
        np.testing.assert_array_equal(mesh.faces,self.f)
        self.assertEqual(mesh.display.GetNumberOfCells(),2)
        self.assertEqual(mesh.display.GetNumberOfPoints(),6)
        self.assertEqual(mesh.texture.name,'sample.jpg')
        target=self.root/'out.vtp'; export_vtp(mesh,np.array([1,0],np.int8),target)
        reader=vtk.vtkXMLPolyDataReader(); reader.SetFileName(str(target)); reader.Update()
        np.testing.assert_array_equal(vtk_to_numpy(reader.GetOutput().GetCellData().GetArray('source_face_id')),[0,1])
        np.testing.assert_array_equal(vtk_to_numpy(reader.GetOutput().GetCellData().GetArray('face_label')),[1,0])
    def test_discovery_deduplicates_prepared_files(self):
        demo=create_demo(self.root/'dataset')
        folder=demo/'SYNTHETIC_DEMO_1'
        atomic_npz(folder/'geometry.npz',vertices=self.v,faces=self.f)
        atomic_npz(folder/'features.npz',features=np.zeros((4,3)))
        atomic_npz(demo/'annotations'/'rater1'/'geometry.npz',vertices=self.v,faces=self.f)
        found=discover(demo); self.assertEqual(len(found),3)
        self.assertTrue(all(p.name=='roi.vtp' for p in found))
        self.assertEqual(len(discover(demo,'DEMO_2')),1)
    def test_atomic_save_and_bad_labels(self):
        store=AnnotationStore(self.root,self.root/'annotations','rater1'); mesh=load_mesh(self.source)
        store.save(mesh,np.array([0,1],np.int8),False,'first')
        store.save(mesh,np.array([1,0],np.int8),True,'second')
        self.assertEqual(store.load(mesh)[2],'second')
        self.assertFalse(list(store.path_for(self.source).parent.glob('*.tmp')))
        with self.assertRaises(ValueError):
            AnnotationStore(self.root,self.root/'annotations','../../bad')

    def test_labeled_ply_roundtrip_binary_and_ascii(self):
        from plyfile import PlyData
        mesh=load_mesh(self.source); labels=np.array([1,-1],np.int8)
        for binary in (True,False):
            target=self.root/f'export_{binary}.ply'
            export_ply(mesh,labels,target,binary=binary)
            restored=load_mesh(target)
            np.testing.assert_array_equal(restored.vertices,mesh.vertices)
            np.testing.assert_array_equal(restored.faces,mesh.faces)
            np.testing.assert_array_equal(restored.imported_labels,labels)
            self.assertEqual(restored.fingerprint,mesh.fingerprint)
            data=PlyData.read(target)
            np.testing.assert_array_equal(data['face'].data['quality'],labels)
            np.testing.assert_array_equal(data['face'].data['source_face_id'],[0,1])
            np.testing.assert_array_equal(data['vertex'].data['red'],255)
        with self.assertRaises(ValueError):export_ply(mesh,labels,self.source)

    def test_roi_preparation_mapping_and_input_unchanged(self):
        atomic_npz(self.source,vertices=self.v,faces=self.f,face_roi_mask=[False,True],rgb=np.tile([20,30,40],(4,1)))
        original=self.source.read_bytes(); mesh=load_mesh(self.source)
        target=self.root/'prepared/roi.ply'; prepare_roi_ply(mesh,target)
        cropped=load_mesh(target)
        self.assertEqual(len(cropped.faces),1); self.assertEqual(len(cropped.vertices),3)
        with np.load(target.with_suffix('.mapping.npz')) as m:
            np.testing.assert_array_equal(cropped.vertices,mesh.vertices[m['source_vertex_id']])
            np.testing.assert_array_equal(m['source_face_id'],[1])
            np.testing.assert_array_equal(m['source_vertex_id'][cropped.faces],mesh.faces[[1]])
        self.assertEqual(self.source.read_bytes(),original)
        with self.assertRaises(FileExistsError):prepare_roi_ply(mesh,target)

    def test_shading_normals_preserve_geometry(self):
        mesh=load_mesh(self.source); rendered=shaded_display(mesh)
        self.assertEqual(rendered.GetNumberOfCells(),len(mesh.faces))
        np.testing.assert_array_equal(vtk_to_numpy(rendered.GetPoints().GetData()),mesh.vertices)
        np.testing.assert_array_equal(vtk_to_numpy(rendered.GetPointData().GetNormals()),np.tile([0,0,1],(4,1)))
        np.testing.assert_array_equal(vtk_to_numpy(rendered.GetCellData().GetNormals()),np.tile([0,0,1],(2,1)))

    def test_texture_ply_export_copies_image_and_preserves_uv(self):
        from mesh_io import Mesh,polydata
        uv=np.array([[0,0],[1,0],[0,1],[1,1]],dtype=np.float32)
        texture=self.root/'input.png'
        texture.write_bytes(b'texture bytes copied unchanged')
        mesh=Mesh(self.source,self.v,self.f,polydata(self.v,self.f,uv=uv),np.ones(2,bool),texture)
        target=self.root/'export/textured.ply'
        export_ply(mesh,np.array([1,0]),target)
        restored=load_mesh(target)
        self.assertEqual(restored.texture.read_bytes(),texture.read_bytes())
        np.testing.assert_array_equal(vtk_to_numpy(restored.display.GetPointData().GetTCoords()),uv[self.f.ravel()])
        np.testing.assert_array_equal(restored.faces,self.f)
        self.assertEqual(shaded_display(restored).GetNumberOfCells(),2)

if __name__=='__main__':
    unittest.main()
