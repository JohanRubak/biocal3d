"""Integration checks against the bundled, existing BioCal3D annotation loader."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import orient_scanners as app
from mesh_io import Mesh,polydata,export_ply,atomic_npz,load_mesh,AnnotationStore
from roi_review_store import read_reviewed
from biocal3d_raster import rasterize
import vtk


def make_fixture(root, real=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if real:
        d=app.arrays(real);v=d['vertices'];f=d['faces'];rgb=d['rgb']
    else:
        v=np.array([[-2.,-2.,0],[2.,-2.,.05],[2.,2.,0],[-2.,2.,0],[0.,0.,.12]])
        f=np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])
        rgb=np.array([[200,30,30],[80,180,30],[40,80,210],[160,140,30],[140,50,170]],np.uint8)
    out=[]
    for i,s in enumerate(app.SCANNERS[:4]):
        rel=Path('samples')/s/'G5'/'BCG5T1-1';folder=root/rel;folder.mkdir(parents=True)
        texture=None
        uv=np.column_stack(((v[:,0]-v[:,0].min())/np.ptp(v[:,0]),(v[:,1]-v[:,1].min())/np.ptp(v[:,1]))) if s=='LABscanner' else None
        if uv is not None:
            from PIL import Image
            texture=folder/'source.jpg';Image.fromarray(np.array([[[220,30,70],[40,90,160]],[[90,130,10],[210,200,50]]],np.uint8)).save(texture)
        m=Mesh(folder/'original.ply',v.copy(),f.copy(),polydata(v,f,rgb,uv),np.ones(len(f),bool),texture)
        m.source_face_ids=np.arange(len(f))+100;m.source_vertex_ids=np.arange(len(v))+200;m.source_domain='raw-sha256:fixture'
        export_ply(m,None,folder/'roi.ply')
        p=polydata(v,f,rgb,uv)
        from vtk.util.numpy_support import numpy_to_vtk
        a=numpy_to_vtk(m.source_face_ids,deep=True);a.SetName('RawTriangleId');p.GetCellData().AddArray(a)
        a=numpy_to_vtk(m.source_vertex_ids,deep=True);a.SetName('RawVertexId');p.GetPointData().AddArray(a)
        w=vtk.vtkXMLPolyDataWriter();w.SetFileName(str(folder/'roi.vtp'));w.SetInputData(p);w.Write()
        t=app.rotation(i*7);t[:3,3]=[1,2,3]
        geo=dict(vertices=v,faces=f,vertex_rgb=rgb.astype(float)/255,normals=np.tile([0,0,1.],(len(v),1)),
                 vertices_raw=app.points(v,np.linalg.inv(t)),raw_triangle_id=m.source_face_ids,raw_vertex_id=m.source_vertex_ids,
                 original_to_canonical=t,canonical_to_original=np.linalg.inv(t),source_ply='unavailable.ply',source_sha256='fixture',
                 extraction_json=np.array(json.dumps({'settings':{'cx':0.,'cy':0.,'z':0.,'radius':6.,'tilt_x':2.,'tilt_y':3.}})))
        if uv is not None:geo.update(uv=uv,source_texture=str(texture))
        atomic_npz(folder/'geometry.npz',**geo)
        fov=max(14.,float(np.abs(v[:,:2]).max()*2.5));size=32
        tex=np.asarray(Image.open(texture).convert('RGB'),dtype=np.float32)/255 if texture else None
        maps=rasterize({'vertices':v,'faces':f,'rgb':rgb.astype(float)/255,'uv':uv,'texture':tex},v,size,fov)
        atomic_npz(folder/'maps.npz',**maps,vertices=v,faces=f,vertices_raw=geo['vertices_raw'],origin=np.zeros(3),basis=np.eye(3),fov_mm=fov,
                   original_to_canonical=t,canonical_to_original=np.linalg.inv(t),mm_per_pixel=fov/size)
        app.dump(folder/'metadata.json',{'scanner':s,'geometry_review_status':'approved','color_source':'texture_uv' if uv is not None else 'vertex_rgb'})
        base=load_mesh(folder/'roi.ply');ver='fixture_review'
        history=root/'annotations'/'_roi_review'/rel/'roi.ply'
        atomic_npz(history/'versions'/ver/'surface.npz',vertices=v,faces=f,face_roi_mask=np.ones(len(f),bool),
                   rgb=rgb,source_face_ids=m.source_face_ids,source_vertex_ids=m.source_vertex_ids,source_domain=np.array(m.source_domain))
        record={'version':ver,'status':'approved','input_fingerprint':base.fingerprint,'settings':{'cx':0.,'cy':0.,'z':0.,'radius':6.,'tilt_x':2.,'tilt_y':3.}}
        app.dump(history/'current.json',record);app.dump(history/'versions'/ver/'review.json',record)
        labels=(np.arange(len(f))%3-1).astype(np.int8)
        effective=read_reviewed(AnnotationStore(root,root/'annotations','bruna'),folder/'roi.ply')
        for rater in ('bruna','katrin'):
            atomic_npz(root/'annotations'/rater/rel/'roi.ply.labels.npz',face_labels=labels,active_faces=effective.active,
                       source_face_ids=m.source_face_ids,source_domain=np.array(m.source_domain),face_indices=np.arange(len(f)),
                       geometry_review_version=np.array(ver),geometry_review_status=np.array('approved'),fingerprint=np.array(effective.fingerprint),
                       source_relative=np.array((rel/'roi.ply').as_posix()),completed=np.array(True),notes=np.array('Keep these notes'),
                       annotator=np.array(rater),camera=np.array(json.dumps({'position':[0,0,30],'focal_point':[0,0,0],'view_up':[0,1,0]})))
        row={'scanner':s,'sample_id':'BCG5T1-1','sample_dir':rel.as_posix(),'annotation_mesh':(rel/'roi.ply').as_posix(),
             'scan_id':s+'__BCG5T1-1','ply_sha256':app.sha(folder/'roi.ply'),'split':'train','study_phase':'agreement'}
        out.append(row)
    for name in ('scan_manifest.csv','annotation_inventory.csv'):app.write_rows(root/name,out)
    app.dump(root/'dataset_index.json',{'scans':out,'config':{}})
    c={'schema':'biocal3d_scanner_orientation_v1','config_id':'fixture','reference':'TRIOS5','frame_evidence':app.frame_evidence(root,out),
       'corrections':{s:{'angle_deg':a,'matrix':app.rotation(a).tolist(),'reviewed':True} for s,a in zip(app.SCANNERS[:4],[0,90,-45,135])}}
    return c,out


class OrientationTests(unittest.TestCase):
    def test_labeled_snapshot_and_maps(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'source';dest=Path(td)/'rotated';c,rr=make_fixture(root)
            before={p:app.sha(p) for p in app.included_files(root)}
            app.apply_dataset(root,dest,c,lambda s:None)
            self.assertEqual({p:app.sha(p) for p in before},before)
            for r in rr:
                src=root/r['sample_dir'];out=dest/r['sample_dir'];t=np.asarray(c['corrections'][r['scanner']]['matrix'])
                a,b=app.arrays(src/'geometry.npz'),app.arrays(out/'geometry.npz')
                np.testing.assert_allclose(b['vertices'],app.points(a['vertices'],t),atol=1e-12)
                np.testing.assert_array_equal(a['vertices_raw'],b['vertices_raw'])
                np.testing.assert_allclose(app.points(b['vertices_raw'],b['original_to_canonical']),b['vertices'],atol=1e-12)
                for rater in ('bruna','katrin'):
                    store=AnnotationStore(dest,dest/'annotations',rater)
                    m=read_reviewed(store,dest/r['annotation_mesh'])
                    labels,done,notes=store.load(m)
                    self.assertTrue(done);self.assertEqual(notes,'Keep these notes')
                    np.testing.assert_array_equal(labels,np.arange(len(m.faces))%3-1)
                    saved=app.arrays(store.path_for(dest/r['annotation_mesh']))
                    self.assertEqual(str(saved['camera'].item()),'{}')
                    self.assertIn('orientation_previous_camera',saved)
                maps=app.arrays(out/'maps.npz');valid=maps['valid'];f=b['faces'][maps['triangle_id'][valid]]
                xyz=np.einsum('ni,nij->nj',maps['barycentric'][valid],b['vertices'][f])
                projected=(xyz-maps['origin'])@maps['basis'];n=maps['valid'].shape[0];row,col=np.nonzero(valid)
                np.testing.assert_allclose(projected[:,0],((col+.5)/n-.5)*maps['fov_mm'],atol=2e-6)
                np.testing.assert_allclose(projected[:,1],(.5-(row+.5)/n)*maps['fov_mm'],atol=2e-6)
            app.load_config(dest/'orientation_corrections_used.json')
            with self.assertRaisesRegex(ValueError,'already been oriented'):app.preflight(dest,c)

    def test_stale_labels_and_wrong_frames_block(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'source';c,rr=make_fixture(root);g=root/rr[0]['sample_dir']/'geometry.npz'
            d=app.arrays(g);d['original_to_canonical'][0,3]+=1;atomic_npz(g,**d)
            with self.assertRaisesRegex(ValueError,'Coordinate frame differs'):app.preflight(root,c)
            d['original_to_canonical'][0,3]-=1;atomic_npz(g,**d)
            p=next((root/'annotations'/'bruna').rglob('*.labels.npz'));d=app.arrays(p);d['fingerprint']=np.array('invalid');atomic_npz(p,**d)
            dest=Path(td)/'out'
            with self.assertRaisesRegex(ValueError,'fingerprint mismatch'):app.apply_dataset(root,dest,c,lambda s:None)
            self.assertFalse(dest.exists())

    def test_crop_normal_rotates(self):
        d={'cx':1.,'cy':2.,'z':3.,'tilt_x':10.,'tilt_y':20.}
        n=np.array([np.tan(np.deg2rad(20)),-np.tan(np.deg2rad(10)),1.])
        for a in (-90,20,180):
            t=app.rotation(a);b=app.extraction(d,t)
            actual=np.array([np.tan(np.deg2rad(b['tilt_y'])),-np.tan(np.deg2rad(b['tilt_x'])),1.])
            np.testing.assert_allclose(actual,app.directions(n,t),atol=1e-12)

    def test_ply_face_properties_are_preserved(self):
        from plyfile import PlyData
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'source';c,rr=make_fixture(root)
            p=root/rr[-1]['annotation_mesh'];m=load_mesh(p)
            out=p.with_name('export.labeled.ply')
            labels=(np.arange(len(m.faces))%3-1).astype(np.int8)
            export_ply(m,labels,out)
            before=PlyData.read(str(out),mmap=False)
            original={e.name:{k:np.stack(e.data[k]) if e.data[k].dtype.kind=='O' else e.data[k].copy() for k in e.data.dtype.names} for e in before.elements}
            app.transform_mesh(out,app.rotation(73))
            after=PlyData.read(str(out),mmap=False)
            for e in after.elements:
                for k in e.data.dtype.names:
                    if e.name=='vertex' and k in ('x','y','z','nx','ny','nz'):continue
                    values=np.stack(e.data[k]) if e.data[k].dtype.kind=='O' else e.data[k]
                    np.testing.assert_array_equal(values,original[e.name][k])

    def test_texture_v_convention_is_recovered(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'source';c,rr=make_fixture(root)
            row=next(r for r in rr if r['scanner']=='LABscanner');folder=root/row['sample_dir']
            g=app.arrays(folder/'geometry.npz');old=app.arrays(folder/'maps.npz')
            tex=np.asarray(Image.open(folder/'source.jpg').convert('RGB'),dtype=np.float32)/255
            old.update(rasterize({'vertices':g['vertices'],'faces':g['faces'],'rgb':g['vertex_rgb'],'uv':g['uv'],'texture':tex},g['vertices'],32,float(old['fov_mm']),False))
            atomic_npz(folder/'maps.npz',**old)
            out=Path(td)/'output';app.apply_dataset(root,out,c,lambda s:None)
            maps=app.arrays(out/row['sample_dir']/'maps.npz');geo=app.arrays(out/row['sample_dir']/'geometry.npz')
            valid=maps['valid'];uv=np.einsum('ni,nij->nj',maps['barycentric'][valid],geo['uv'][geo['faces'][maps['triangle_id'][valid]]])
            from biocal3d_raster import sample_texture
            np.testing.assert_allclose(maps['rgb'][valid],sample_texture(tex,uv,False),atol=1e-6)

    def test_calibration_agreement_parent(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'study'
            c,_=make_fixture(root/'calibration')
            make_fixture(root/'agreement')
            out=Path(td)/'oriented_study'
            app.apply_tree(root,out,c,lambda s:None)
            for phase in ('calibration','agreement'):
                result=json.loads((out/phase/app.STAMP).read_text())
                self.assertEqual(result['annotation_files_verified'],8)
                self.assertEqual(result['status'],'complete')
            self.assertEqual(json.loads((out/'orientation_collection.json').read_text())['status'],'complete')

if __name__=='__main__':unittest.main()
