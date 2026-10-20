"""Portable pilot copies: frozen approvals, clean ratings, relocation and analysis."""
import csv,json,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import vtk
from PIL import Image
from annotator.mesh_io import AnnotationStore,Mesh,polydata,load_mesh,export_ply,discover,mesh_appearance
from annotator.roi_review_store import approve_displayed_surface,read_reviewed,read_state,review_dir,save_review,digest
from roi_geometry import Extraction
from labeling_study import create_study_folders,write_csv,inventory
from analyse_labeling_agreement import analyse


def fixture(root,specimens=8):
    root.mkdir();rows=[]
    v=np.array([[-1.,-1,0],[1,-1,0],[-1,1,0],[1,1,0]])
    f=np.array([[0,1,2],[1,3,2]])
    texture=root/'original_texture.png';Image.new('RGB',(4,4),(190,40,110)).save(texture)
    for group,count in ((5,specimens),(4,1)):
        for specimen in range(1,count+1):
            for scanner in ('TRIOS3','TRIOS5','LABscanner','iTERO'):
                for time in range(3):
                    name=f'BCG{group}T{time}-{specimen}';folder=root/'samples'/scanner/f'G{group}'/name;folder.mkdir(parents=True)
                    # A previously corrected input has fewer faces. Copies must keep it exactly.
                    faces=f[:1] if specimen==1 and time==1 else f
                    uv=np.array([[0.,0],[1,0],[0,1],[1,1]]) if scanner in ('LABscanner','iTERO') else None
                    rgb=np.tile([190,40,110],(4,1)).astype(np.uint8)
                    mesh=Mesh(root/'original.ply',v,faces,polydata(v,faces,rgb,uv),np.ones(len(faces),bool),texture if uv is not None else None)
                    mesh.source_face_ids=np.arange(len(faces))+10
                    export_ply(mesh,None,folder/'roi.ply')
                    writer=vtk.vtkXMLPolyDataWriter();writer.SetFileName(str(folder/'roi.vtp'));writer.SetInputData(mesh.display);writer.Write()
                    np.savez_compressed(folder/'geometry.npz',vertices=v,faces=faces,raw_triangle_id=mesh.source_face_ids,raw_vertex_id=np.arange(4),source_ply=str(root/'raw.ply'),roi_radius=2.)
                    row=dict(scan_id=scanner+'__'+name,physical_id=f'G{group}_S{specimen}',group=group,scanner=scanner,
                             timepoint=f'T{time}',time_idx=time,stage=('baseline','partial','clean')[time],
                             annotation_mesh=str((folder/'roi.ply').relative_to(root)),sample_dir=str(folder.relative_to(root)),ply_sha256=digest(folder/'roi.ply'))
                    (folder/'metadata.json').write_text(json.dumps(row))
                    rows.append(row)
                    store=AnnotationStore(root,root/'annotations','johan');approve_displayed_surface(store,load_mesh(folder/'roi.ply'))
    write_csv(root/'annotation_inventory.csv',rows);write_csv(root/'scan_manifest.csv',rows)
    (root/'geometry_rebuild.json').write_text(json.dumps(dict(status='complete',scans=rows)))
    # Deliberately add old training ratings and a stray exported file. Neither may enter a new study.
    mesh=read_reviewed(store,root/rows[0]['annotation_mesh'])
    for rater in ('johan','bruna'):
        AnnotationStore(root,root/'annotations',rater).save(mesh,np.ones(len(mesh.faces),np.int8),True,'Old rating')
    (mesh.path.parent/'roi.ply.labels.npz').write_bytes(b'not study input')
    return rows


class PortableStudyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.home=Path(self.tmp.name);self.root=self.home/'approved snapshot'
    def tearDown(self):self.tmp.cleanup()

    def test_full_g5_counts_empty_labels_relocation_and_agreement(self):
        rows=fixture(self.root);before={str(p.relative_to(self.root)):digest(p) for p in self.root.rglob('*') if p.is_file()}
        counts=create_study_folders(self.root)
        self.assertEqual(counts,dict(calibration=dict(specimens=2,scans=24),agreement=dict(specimens=6,scans=72)))
        output=self.home/'G5_labeling_study';plan=json.loads((output/'study_plan.json').read_text())
        self.assertEqual(plan['status'],'complete');self.assertFalse(plan['plaque_labels_copied'])
        phases={};expected={r['scan_id']:r for r in rows};source_store=AnnotationStore(self.root,self.root/'annotations','johan')
        for phase in ('calibration','agreement'):
            stage=output/phase;copied=inventory(stage);phases[phase]={r['physical_id'] for r in copied}
            self.assertEqual(len(discover(stage)),counts[phase]['scans'])
            self.assertEqual({p.name for p in (stage/'annotations').iterdir()},{'_roi_review','bruna','katrin'})
            for row in copied:
                self.assertEqual(row['group'],'5');original=read_reviewed(source_store,self.root/expected[row['scan_id']]['annotation_mesh'])
                store=AnnotationStore(stage,stage/'annotations','bruna');mesh=read_reviewed(store,stage/row['annotation_mesh'])
                self.assertEqual(mesh.fingerprint,original.fingerprint);self.assertEqual(mesh.review_version,original.review_version)
                self.assertEqual(mesh.review_status,'approved');np.testing.assert_array_equal(mesh.source_face_ids,original.source_face_ids)
                for a,b in zip(mesh_appearance(mesh),mesh_appearance(original)):
                    if a is None:self.assertIsNone(b)
                    else:np.testing.assert_array_equal(a,b)
                labels,done,_=store.load(mesh);self.assertFalse(done);self.assertTrue((labels==-2).all())
                if mesh.texture:self.assertTrue(mesh.texture.is_relative_to(stage))
                self.assertEqual((review_dir(store,mesh.path)/'current.json').read_bytes(),(review_dir(source_store,original.path)/'current.json').read_bytes())
                self.assertEqual((review_dir(store,mesh.path)/'versions'/mesh.review_version/'surface.npz').read_bytes(),
                                 (review_dir(source_store,original.path)/'versions'/original.review_version/'surface.npz').read_bytes())
            self.assertFalse(any(stage.rglob('*.labels.npz')))
        self.assertFalse(phases['calibration']&phases['agreement'])
        self.assertEqual(before,{str(p.relative_to(self.root)):digest(p) for p in self.root.rglob('*') if p.is_file()})
        relocated=self.home/'shared drive'/'Pilot';relocated.parent.mkdir();shutil.move(output,relocated)
        shutil.move(self.root,self.home/'source unavailable')
        stage=relocated/'agreement'
        for row in inventory(stage):
            for name in ('bruna','katrin'):
                store=AnnotationStore(stage,stage/'annotations',name);mesh=read_reviewed(store,stage/row['annotation_mesh'])
                if mesh.display.GetPointData().GetTCoords() is not None:
                    self.assertIsNotNone(mesh.texture);self.assertTrue(mesh.texture.is_relative_to(stage))
                store.save(mesh,np.ones(len(mesh.faces),np.int8),True,'Independent test')
        result=analyse(stage,stage/'labeling_plan',rater1='bruna',rater2='katrin')
        self.assertEqual(result['complete_pairs'],72);self.assertEqual(result['missing_or_failed'],0)
        self.assertEqual(result['all_stages']['plaque_dice']['mean_of_specimen_means'],1.)

    def test_preflight_incomplete_pending_and_changed_since_rebuild(self):
        rows=fixture(self.root,2);dest=self.home/'study';kwargs=dict(output=dest,calibration=1,agreement=1)
        (self.root/'INCOMPLETE_REBUILD.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'successfully REBUILT'):create_study_folders(self.root,**kwargs)
        (self.root/'INCOMPLETE_REBUILD.json').unlink()
        store=AnnotationStore(self.root,self.root/'annotations','johan');source=self.root/rows[0]['annotation_mesh']
        state_path=review_dir(store,source)/'current.json';original=state_path.read_bytes();state=json.loads(original);state['status']='pending';state_path.write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError,'not ROI-approved'):create_study_folders(self.root,**kwargs)
        state_path.write_bytes(original)
        mesh=read_reviewed(store,source)
        save_review(store,source,mesh,np.array([True,False]),Extraction(radius=2.),'approved','Later correction',mesh.review_version)
        with self.assertRaisesRegex(ValueError,'changed after rebuilding'):create_study_folders(self.root,**kwargs)
        self.assertFalse(dest.exists())

    def test_existing_study_not_overwritten_and_selection_reproducible(self):
        fixture(self.root,2);first=self.home/'first';second=self.home/'second'
        create_study_folders(self.root,first,calibration=1,agreement=1)
        with self.assertRaisesRegex(ValueError,'new/empty'):create_study_folders(self.root,first,calibration=1,agreement=1)
        create_study_folders(self.root,second,calibration=1,agreement=1)
        self.assertEqual((first/'assignments.csv').read_bytes(),(second/'assignments.csv').read_bytes())
        with self.assertRaisesRegex(ValueError,'0 physical specimens'):create_study_folders(self.root,self.home/'wronggroup',group=9,calibration=1,agreement=1)

    def test_cli_paths_with_spaces_and_excluded_scans(self):
        rows=fixture(self.root,2);output=self.home/'new study folders'
        command=[sys.executable,str(Path(__file__).with_name('labeling_study.py')),'folders','--root',str(self.root),'--output',str(output),'--calibration','1','--agreement','1']
        completed=subprocess.run(command,capture_output=True,text=True)
        self.assertEqual(completed.returncode,0,completed.stderr);self.assertTrue((output/'study_plan.json').exists())
        write_csv(self.root/'annotation_inventory.csv',rows[1:])
        with self.assertRaisesRegex(ValueError,'excluded from annotation_inventory'):create_study_folders(self.root,self.home/'excluded',calibration=1,agreement=1)


if __name__=='__main__':unittest.main()
