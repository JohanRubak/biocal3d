"""Meaningful data/protocol integration tests; no GPU or model downloads required.
python -m unittest -v test_study
"""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
from study_data import *
from study_report import metadata_table,baseline_controls,rebuild_report
import biocal3d_train_3d_investigations as runner

def fixture(root):
    records=[]
    for scanner in SCANNERS:
        for g in range(1,7):
            for n in range(1,9):
                for t in range(2 if g<3 else 3):
                    sample=f'BCG{g}T{t}-{n}';folder=root/scanner/f'G{g}'/sample;folder.mkdir(parents=True)
                    axis=np.linspace(-6,6,6);xx,yy=np.meshgrid(axis,axis)
                    amp=.1 if t==0 else .04 if t==1 and g>=3 else .015
                    zz=amp*np.sin(xx)*np.cos(yy)+n*.0001*xx
                    v=np.c_[xx.ravel(),yy.ravel(),zz.ravel()].astype('float32');faces=[]
                    for i in range(5):
                        for j in range(5):
                            a=i*6+j;faces.extend([[a,a+1,a+6],[a+1,a+7,a+6]])
                    f=np.array(faces,'int64');rgb=np.tile([.7,.2,.35],(len(v),1)).astype('float32')
                    normals=np.tile([0,0,1.],(len(v),1)).astype('float32')
                    p=folder/'geometry.npz';np.savez_compressed(p,vertices=v,faces=f,normals=normals,vertex_rgb=rgb)
                    records.append({'scan_id':scanner+'__'+sample,'scanner':scanner,'group':g,'physical_id':f'BCG{g}-{n}',
                                    'stage':identity(sample)[-1],'sample_dir':folder.relative_to(root).as_posix(),
                                    'status':'OK','geometry_sha256':digest(p)})
    (root/'dataset_index.json').write_text(json.dumps({'scans':records}),encoding='utf-8')

class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)/'snapshot';fixture(cls.root)
        cls.rows,cls.audit,_=load_snapshot(cls.root);cls.ref=reference_split(cls.rows)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_complete_cohort_and_reference(self):
        self.assertEqual(len(self.rows),512);self.assertEqual(self.audit['physical_specimens'],48)
        self.assertEqual([sum(r['split']==p for r in self.ref) for p in ['train','val','test']],[384,64,64])
        self.assertEqual(len({r['physical_id'] for r in self.ref if r['split']=='val' and r['stage']=='partial'}),4)
        assert_partition(self.ref)
    def test_path_escape_rejected(self):
        root=Path(self.tmp.name)/'escape';root.mkdir();doc={'scans':[{'status':'OK','sample_dir':'../outside','scan_id':'BCG1T0-1'}]}
        (root/'dataset_index.json').write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError,'escapes'):load_snapshot(root)
    def test_hash_mismatch_and_missing_scan_rejected(self):
        root=Path(self.tmp.name)/'hash';root.mkdir();sample=self.rows[0];folder=root/'sample';folder.mkdir()
        import shutil
        shutil.copy(sample['mesh_path'],folder/'geometry.npz')
        r={'status':'OK','sample_dir':'sample','scan_id':sample['scan_id'],'scanner':sample['scanner'],'group':sample['group'],
           'physical_id':sample['physical_id'],'stage':sample['stage'],'geometry_sha256':'wrong'}
        (root/'dataset_index.json').write_text(json.dumps({'scans':[r]}))
        with self.assertRaisesRegex(ValueError,'provenance'):load_snapshot(root)
        r['geometry_sha256']=digest(folder/'geometry.npz');(root/'dataset_index.json').write_text(json.dumps({'scans':[r]}))
        with self.assertRaisesRegex(ValueError,'Cohort incomplete'):load_snapshot(root)
    def test_grouped_cv_and_scanner_holdout_have_no_subject_leakage(self):
        reserved={r['physical_id'] for r in self.ref if r['split']=='test'};assessed=[]
        for k,rows in assessment_folds(self.ref,5,42):
            assert_partition(rows);self.assertFalse(reserved&{r['physical_id'] for r in rows})
            assessed.extend(r['row_id'] for r in rows if r['split']=='assessment')
            for scanner in SCANNERS:
                rr=loso_rows(rows,scanner);assert_partition(rr)
                self.assertTrue(all(r['scanner']==scanner for r in rr if r['split']=='assessment'))
                self.assertTrue(all(r['scanner']!=scanner for r in rr if r['split']!='assessment'))
        expected={r['row_id'] for r in self.ref if r['split']!='test'}
        self.assertEqual(len(assessed),len(set(assessed)));self.assertEqual(set(assessed),expected)
    def test_saved_split_leakage_is_rejected(self):
        rr=[dict(r,source_split='train') for r in self.rows]
        pid=rr[0]['physical_id'];next(r for r in rr[1:] if r['physical_id']==pid)['source_split']='val'
        with self.assertRaisesRegex(ValueError,'leaks'):reference_split(rr)
    def test_area_sampling_has_new_points_and_display_rgb_is_separate(self):
        a=runner.arguments(['--points','256','--suite','core']);j=runner.plan_jobs(self.ref,a)[0]
        r=self.rows[0];s=make_surface(r,j['surface'],'pointnext_s');same=make_surface(r,j['surface'],'pointnext_s')
        np.testing.assert_array_equal(s['coord'],same['coord']);self.assertGreater(len(np.unique(s['coord'],axis=0)),200)
        self.assertTrue(np.allclose(s['barycentric'].sum(1),1));self.assertTrue((s['barycentric']>=0).all())
        self.assertTrue((s['color']==0).all());self.assertTrue(s['display_rgb'].max()>0)
        c=make_surface(r,dict(j['surface'],inputs='geometry_rgb'),'pointnext_s');np.testing.assert_array_equal(c['color'],c['display_rgb'])
    def test_triangle_streams_and_point_controls(self):
        cfg=runner.plan_jobs(self.ref,runner.arguments(['--points','256']))[0]['surface'];r=self.rows[0]
        s=make_surface(r,cfg,'tsgcnet');self.assertEqual(s['tsgc'].shape,(256,24))
        np.testing.assert_allclose(s['tsgc'][:,:9].reshape(-1,3,3).mean(1),s['coord'],atol=1e-6)
        xyz=make_surface(r,dict(cfg,xyz_only=True),'pointnext_s');self.assertTrue((xyz['normal']==0).all());self.assertTrue((xyz['tsgc'][:,12:]==0).all())
        vox=make_surface(r,dict(cfg,surface_grid_mm=1.),'pointnext_s');self.assertTrue((vox['face_index']==-1).all());self.assertTrue(np.isfinite(vox['coord']).all())
        with self.assertRaisesRegex(ValueError,'TSGCNet'):make_surface(r,dict(cfg,surface_grid_mm=1.),'tsgcnet')
    def test_rigid_transform_and_voxel_origin(self):
        cfg=runner.plan_jobs(self.ref,runner.arguments(['--points','64']))[0]['surface'];s=make_surface(self.rows[0],cfg,'tsgcnet')
        d=transformed(s,'translate_scaled_0p1');np.testing.assert_allclose(d['coord']-s['coord'],np.tile([.1,-.1,.05],(64,1)),atol=1e-6)
        np.testing.assert_allclose(d['grid_origin']-s['grid_origin'],[.1,-.1,.05],atol=1e-6)
        d=transformed(s,'rotate_z_90');np.testing.assert_allclose(np.linalg.norm(d['normal'],axis=1),np.linalg.norm(s['normal'],axis=1),atol=1e-6)
    def test_intervals_refuse_one_partial_subject_and_vary_on_new_cohort(self):
        rows=[r for r in self.ref if r['split']=='val'];prob=np.full((len(rows),3),.05)
        for i,r in enumerate(rows):prob[i,STAGES.index(r['stage'])]=.9
        # Introduce different specimen-level errors to exercise resampling uncertainty.
        pid=next(r['physical_id'] for r in rows if r['stage']=='partial')
        for i,r in enumerate(rows):
            if r['physical_id']==pid and r['stage']=='partial':prob[i]=[.9,.05,.05]
        result=cluster_interval(rows,prob,repeats=100);self.assertIsNotNone(result['interval']);self.assertGreater(result['interval'][1],result['interval'][0])
        old=[r for r in rows if r['group']<3 or r['physical_id']==pid];ix=[rows.index(r) for r in old]
        self.assertIsNone(cluster_interval(old,prob[ix],repeats=20)['interval'])
    def test_plans_and_no_torch_audit_entrypoints(self):
        a=runner.arguments(['--suite','full']);jobs=runner.plan_jobs(self.ref,a)
        self.assertEqual(sum(j['protocol']=='reference' for j in jobs),23)
        self.assertEqual(sum(j['protocol']=='cv' for j in jobs),10)
        self.assertEqual(sum(j['protocol']=='loso' for j in jobs),40)
        out=Path(self.tmp.name)/'plan'
        self.assertEqual(runner.main(['--data-root',str(self.root),'--output',str(out),'--plan']),0)
        self.assertTrue((out/'comparison.html').exists())
    def test_control_pipeline_and_reports(self):
        out=Path(self.tmp.name)/'controls';out.mkdir();rows=[r for r in self.ref if r['split']!='test']
        # Full reference protocol with real train-only sklearn preprocessing.
        meta=metadata_table(rows,out);result=baseline_controls(rows,meta,out/'controls',bootstrap=10)
        self.assertEqual(len(result),5)
        self.assertTrue((out/'controls'/'colour_only'/'predictions_val.csv').exists())
        self.assertTrue(all(r['scans']==64 for r in result))
        rebuild_report(out);self.assertTrue((out/'comparison.html').exists())

if __name__=='__main__':unittest.main(verbosity=2)
