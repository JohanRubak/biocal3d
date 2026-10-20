"""Synthetic regression/integration tests; these are not model performance tests."""
from pathlib import Path
import contextlib, csv, io, json, shutil, tempfile, unittest
from unittest.mock import patch
import numpy as np
import followup as runner
import followup_models as fm
from approved_data import interior_mask
from extensions import output_lock, stage_pairs, rebuild_extensions, csv_rows
from followup_report import rebuild
from study_data import *
from study_report import predictions


def fixture(root):
    records=[]; reviews=[]
    axis=np.linspace(-5,5,11);xx,yy=np.meshgrid(axis,axis)
    vertices=np.c_[xx.ravel(),yy.ravel(),np.zeros(xx.size)].astype('float32')
    faces=[]
    for i in range(10):
        for j in range(10):
            a=i*11+j;faces.extend([[a,a+1,a+11],[a+1,a+12,a+11]])
    faces=np.array(faces,'int64')
    rgb=np.c_[np.linspace(.1,.8,len(vertices)),np.full(len(vertices),.25),np.full(len(vertices),.6)].astype('float32')
    for scanner in SCANNERS:
        for g in range(1,7):
            for n in range(1,9):
                for t in range(2 if g<3 else 3):
                    sample=f'BCG{g}T{t}-{n}';folder=root/'samples'/scanner/f'G{g}'/sample;folder.mkdir(parents=True)
                    p=folder/'geometry.npz'
                    np.savez_compressed(p,vertices=vertices,faces=faces,vertex_rgb=rgb,roi_radius=6.,
                        extraction_json=np.array(json.dumps({'settings':dict(cx=0,cy=0,z=0,radius=6.,tilt_x=0,tilt_y=0)})),
                        source_ply=np.array(r'C:\old_pc\does_not_exist.ply'),source_texture=np.array(r'C:\old_pc\missing.jpg'))
                    scan_id=scanner+'__'+sample
                    records.append(dict(scan_id=scan_id,scanner=scanner,group=g,timepoint=f'T{t}',time_idx=t,
                        physical_id=f'G{g}_S{n}',stage=identity(sample)[-1],sample_dir=folder.relative_to(root).as_posix(),
                        status='OK',geometry_sha256=digest(p),geometry_review_status='approved',split='unassigned'))
                    reviews.append(dict(scan_id=scan_id,status='approved'))
    write_json(root/'dataset_index.json',dict(schema_version=3,scans=records))
    write_json(root/'geometry_rebuild.json',dict(status='complete',scans=reviews))


class ApprovedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.base=Path(cls.tmp.name)
        cls.data=cls.base/'biocal3d'/'data'/'Approved ROIs';fixture(cls.data)
        cls.output=cls.base/'outputs'
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def args(self,*extra):
        return runner.arguments(['--data-root',str(self.data),'--output',str(self.output),*extra])
    def rows(self):return runner.load_matched(self.args())[0]

    def test_counts_and_independent_specimen_protocols(self):
        rows=self.rows();a=self.args()
        self.assertEqual({p:sum(r['split']==p for r in rows) for p in ['train','val','test']},dict(train=384,val=64,test=64))
        self.assertEqual(len({r['physical_id'] for r in rows}),48)
        reserved={r['physical_id'] for r in rows if r['split']=='test'}
        for action,count in [('baseline',23),('colour',36),('interior',18),('cv',20),('loso',80)]:
            jobs=runner.plan(rows,a,action);self.assertEqual(len(jobs),count)
            assessment={}
            for job in jobs:
                assert_partition(job['rows']);self.assertFalse(reserved&{r['physical_id'] for r in job['rows']})
                if action=='loso':
                    scanner=job['tag'].split('_',1)[1]
                    self.assertTrue(all(r['scanner']==scanner for r in job['rows'] if r['split']=='assessment'))
                    self.assertTrue(all(r['scanner']!=scanner for r in job['rows'] if r['split']!='assessment'))
                if action in ['cv','loso']:
                    assessment.setdefault((job['model'],job['variant']),[]).extend(r['row_id'] for r in job['rows'] if r['split']=='assessment')
            for ids in assessment.values():self.assertEqual(len(ids),448);self.assertEqual(len(set(ids)),448)

    def test_snapshot_portable_without_raw_files_or_previous_results(self):
        self.assertEqual(len(self.rows()),512)
        row=self.rows()[0];self.assertTrue(check_surface_rgb(row))
        with tempfile.TemporaryDirectory() as temp:
            moved=Path(temp)/'relocated';shutil.copytree(self.data,moved)
            a=self.args('--data-root',str(moved));rows,_,_=runner.load_matched(a)
            self.assertTrue(all(Path(r['mesh_path']).is_relative_to(moved) for r in rows))

    def test_incomplete_unapproved_and_tampered_data_rejected(self):
        marker=self.data/'INCOMPLETE_REBUILD.json';marker.write_text('{}')
        try:
            with self.assertRaisesRegex(ValueError,'INCOMPLETE'):self.rows()
        finally:marker.unlink()
        p=self.data/'dataset_index.json';original=p.read_bytes();doc=json.loads(original)
        try:
            doc['scans'][0]['geometry_review_status']='pending';write_json(p,doc)
            with self.assertRaisesRegex(ValueError,'approved'):self.rows()
            doc['scans'][0]['geometry_review_status']='approved';doc['scans'][0]['geometry_sha256']='bad';write_json(p,doc)
            with self.assertRaisesRegex(ValueError,'provenance'):self.rows()
        finally:p.write_bytes(original)

    def test_source_split_optional_and_explicit_split(self):
        with self.assertRaisesRegex(ValueError,'complete saved'):runner.load_matched(self.args('--split-policy','source'))
        rows=self.rows();p=self.base/'explicit.json'
        write_json(p,{'physical_ids':{part:sorted({r['physical_id'] for r in rows if r['split']==part}) for part in ['train','val','test']}})
        self.assertEqual([(r['row_id'],r['split']) for r in rows],[(r['row_id'],r['split']) for r in runner.load_matched(self.args('--split-json',str(p)))[0]])

    def test_interior_uses_reviewed_frame_and_preserves_source_faces(self):
        row=self.rows()[0];v,f,*_=geometry(row);mask=interior_mask(row,v,f,.8)
        self.assertTrue(mask.any());self.assertTrue((~mask).any())
        config=runner.base_surface(self.args());config['interior_fraction']=.8
        data=make_surface(row,config,'pointnext_s')
        self.assertTrue(mask[data['face_index']].all())
        np.testing.assert_allclose((v[f[data['face_index']]]*data['barycentric'][:,:,None]).sum(1),
            data['coord']*config['scale_mm']+data['center_mm'],atol=1e-5)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'shifted.npz';np.savez(path,extraction_json=np.array(json.dumps({'settings':dict(cx=10,cy=20,z=0,radius=6)})))
            np.testing.assert_array_equal(interior_mask(dict(mesh_path=path,scan_id='shifted'),v+[10,20,0],f,.8),mask)

    def test_mature_pilot_and_matched_colour_statistics(self):
        rows=self.rows();a=self.args('--mature-only','--fold-indices','0')
        self.assertEqual(len(runner.plan(rows,a,'cv')),4);self.assertEqual(len(runner.plan(rows,a,'loso')),16)
        self.assertTrue(all(r['group']>=3 for j in runner.plan(rows,a,'baseline') for r in j['rows']))
        s=make_surface(rows[0],runner.base_surface(a),'pointnext_s');changed=fm.alter_surface(s,'rgb_shuffled','x')
        np.testing.assert_allclose(fm.colour_statistics_numpy(s['color']),fm.colour_statistics_numpy(changed['color']),atol=1e-5)
        np.testing.assert_array_equal(s['coord'],changed['coord'])

    def test_inference_perturbations_preserve_spatial_mapping(self):
        row=self.rows()[0];data=make_surface(row,runner.base_surface(self.args()),'pointnext_s')
        original=data['coord'].copy();rotated=transformed(data,'rotate_z_180')
        np.testing.assert_allclose(rotated['coord'][:,:2],-original[:,:2],atol=1e-7)
        reordered=transformed(data,'point_order')
        v,f,*_=geometry(row)
        np.testing.assert_allclose((v[f[reordered['face_index']]]*reordered['barycentric'][:,:,None]).sum(1),
            reordered['coord']*10+reordered['center_mm'],atol=1e-5)
        shuffled=transformed(data,'shuffle_rgb')
        np.testing.assert_allclose(fm.colour_statistics_numpy(data['color']),fm.colour_statistics_numpy(shuffled['color']),atol=1e-5)
        np.testing.assert_array_equal(data['coord'],original)

    def test_output_identity_rejects_changed_data_or_fitting_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            a=self.args('--output',temp);rows,audit,_=runner.load_matched(a);runner.save_identity(a,rows,audit)
            a.lr*=2
            with self.assertRaisesRegex(ValueError,'different'):runner.save_identity(a,rows,audit)
        with self.assertRaisesRegex(ValueError,'separate'):self.args('--output',str(self.data/'runs'))

    def test_lock_releases_after_errors_and_rejects_concurrent_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,'stop'):
                with output_lock(temp):
                    with self.assertRaisesRegex(RuntimeError,'Another'): 
                        with output_lock(temp):pass
                    raise ValueError('stop')
            self.assertFalse((Path(temp)/'.active_run.lock').exists())

    def test_stage_pairs_known_order_and_reversal(self):
        pp=[dict(scanner='iTERO',physical_id='BCG3-1',truth=i,p_partial=p[1],p_clean=p[2]) for i,p in enumerate([[.9,.05,.05],[.1,.8,.1],[.05,.05,.9]])]
        pairs=stage_pairs(pp);self.assertEqual(len(pairs),3);self.assertTrue(all(x['delta']>0 for x in pairs))
        pp[-1].update(p_partial=0,p_clean=0)
        self.assertEqual(sum(x['reversed'] for x in stage_pairs(pp)),2)

    def test_plan_integration_writes_all_stages_without_opening_mesh_arrays(self):
        with tempfile.TemporaryDirectory() as temp, patch('numpy.load',side_effect=AssertionError('planning opened arrays')):
            with contextlib.redirect_stdout(io.StringIO()):runner.main(['--data-root',str(self.data),'--output',temp,'--plan'])
            out=Path(temp);self.assertTrue((out/'comparison.html').exists())
            self.assertEqual(len(list((out/'partitions').glob('*.csv'))),28)
            self.assertFalse((out/'.active_run.lock').exists())

    def test_audit_excludes_reserved_arrays_and_does_not_change_input(self):
        rows=self.rows();reserved={str(Path(r['mesh_path']).resolve()) for r in rows if r['split']=='test'}
        before={str(p):digest(p) for p in self.data.rglob('*') if p.is_file()}
        original_load=np.load
        def guarded_load(file,*args,**kwargs):
            if isinstance(file,(str,Path)) and str(Path(file).resolve()) in reserved:
                raise AssertionError('Reserved test arrays opened during audit')
            return original_load(file,*args,**kwargs)
        with tempfile.TemporaryDirectory() as temp, patch('numpy.load',side_effect=guarded_load):
            with contextlib.redirect_stdout(io.StringIO()):
                runner.main(['--data-root',str(self.data),'--output',temp,'--action','audit','--skip-controls','--bootstrap','2'])
            self.assertEqual(len(csv_rows(Path(temp)/'qc'/'interior_roi.csv')),448)
        self.assertEqual(before,{str(p):digest(p) for p in self.data.rglob('*') if p.is_file()})

    def test_oof_completion_and_interior_summary_are_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            a=self.args('--output',temp);rows,audit,_=runner.load_matched(a);runner.save_identity(a,rows,audit)
            jobs=[j for j in runner.plan(rows,a,'cv') if j['model']=='pointnext_s' and j['variant']=='stats_only']
            for i,job in enumerate(jobs):
                run=a.output/'runs'/job['name'];rr=[r for r in job['rows'] if r['split']=='assessment'];prob=np.tile([.2,.3,.5],(len(rr),1))
                write_json(run/'config.json',{k:v for k,v in job.items() if k not in ['rows','name']});write_json(run/'status.json',{'status':'complete'})
                write_json(run/'metrics_assessment.json',score(rr,prob));write_csv(run/'predictions_assessment.csv',predictions(rr,prob));rebuild(a.output)
                summary=csv_rows(a.output/'followup_out_of_fold.csv')
                self.assertEqual(summary[0]['complete'],'True' if i==4 else 'False')
            for action,value in [('colour',.8),('interior',.4)]:
                job=next(j for j in runner.plan(rows,a,action) if j['model']=='pointnext_s' and j['variant']=='rgb_combined' and j['seed']==42)
                run=a.output/'runs'/job['name'];rr=[r for r in job['rows'] if r['split']=='val'];prob=np.tile([.2,.3,.5],(len(rr),1));m=score(rr,prob);m['specimen_balanced_accuracy']=value
                write_json(run/'config.json',{k:v for k,v in job.items() if k not in ['rows','name']});write_json(run/'status.json',{'status':'complete'})
                write_json(run/'metrics_val.json',m);write_csv(run/'predictions_val.csv',predictions(rr,prob))
            rebuild(a.output);rebuild_extensions(a.output)
            color=csv_rows(a.output/'colour_pathway_summary.csv')
            interior=csv_rows(a.output/'interior_comparison.csv')
            self.assertAlmostEqual(float(color[0]['BA_mean']),.8);self.assertAlmostEqual(float(interior[0]['delta_interior_minus_full']),-.4)


if __name__=='__main__':unittest.main(verbosity=2)
