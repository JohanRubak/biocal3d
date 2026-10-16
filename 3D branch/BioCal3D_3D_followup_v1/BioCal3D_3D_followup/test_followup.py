"""Synthetic follow-up checks; no benchmark results are inferred from fixtures."""
import importlib.util,json,tempfile,unittest,csv
from pathlib import Path
import numpy as np
from study_data import *
from study_report import predictions
import followup as runner
import followup_models as fm
from followup_report import rebuild

def fixture(root):
    records=[];axis=np.linspace(-6,6,6);xx,yy=np.meshgrid(axis,axis)
    v=np.c_[xx.ravel(),yy.ravel(),np.zeros(xx.size)].astype('float32');f=[]
    for i in range(5):
        for j in range(5):
            a=i*6+j;f.extend([[a,a+1,a+6],[a+1,a+7,a+6]])
    f=np.array(f,'int64');rgb=np.c_[np.linspace(.2,.9,len(v)),np.full(len(v),.2),np.full(len(v),.35)].astype('float32')
    for scanner in SCANNERS:
        for g in range(1,7):
            for n in range(1,9):
                for t in range(2 if g<3 else 3):
                    sample=f'BCG{g}T{t}-{n}';folder=root/scanner/f'G{g}'/sample;folder.mkdir(parents=True)
                    p=folder/'geometry.npz';np.savez_compressed(p,vertices=v,faces=f,vertex_rgb=rgb)
                    records.append(dict(scan_id=scanner+'__'+sample,scanner=scanner,group=g,physical_id=f'BCG{g}-{n}',stage=identity(sample)[-1],sample_dir=folder.relative_to(root).as_posix(),status='OK',geometry_sha256=digest(p),split='unassigned'))
    write_json(root/'dataset_index.json',{'scans':records})

class FollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);fixture(cls.root/'data');raw,_,_=load_snapshot(cls.root/'data');cls.rows=reference_split(raw);cls.old=cls.root/'original'
        write_json(cls.old/'reference_split.json',{'physical_ids':{p:sorted({r['physical_id'] for r in cls.rows if r['split']==p}) for p in ['train','val','test']}});write_csv(cls.old/'scan_manifest.csv',cls.rows)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def args(self,*extra):return runner.arguments(['--original-output',str(self.old),'--data-root',str(self.root/'data'),'--output',str(self.root/'new'),*extra])
    def test_default_counts_and_no_subject_scanner_leakage(self):
        a=self.args();self.assertEqual([len(runner.plan(self.rows,a,x)) for x in ['colour','cv','loso']],[36,20,80]);reserved={r['physical_id'] for r in self.rows if r['split']=='test'}
        for action in ['colour','cv','loso']:
            seen={}
            for job in runner.plan(self.rows,a,action):
                rr=job['rows'];assert_partition(rr);self.assertFalse(reserved & {r['physical_id'] for r in rr})
                if action=='loso':
                    scanner=job['tag'].split('_',1)[1];self.assertTrue(all(r['scanner']==scanner for r in rr if r['split']=='assessment'));self.assertTrue(all(r['scanner']!=scanner for r in rr if r['split']!='assessment'))
                if action!='colour':seen.setdefault((job['model'],job['variant']),[]).extend(r['row_id'] for r in rr if r['split']=='assessment')
            for ids in seen.values():self.assertEqual(len(ids),448);self.assertEqual(len(ids),len(set(ids)))
    def test_mature_only_and_fold_pilot(self):
        a=self.args('--mature-only');self.assertTrue(all(r['group']>=3 for j in runner.plan(self.rows,a,'colour') for r in j['rows']))
        a=self.args('--fold-indices','0');self.assertEqual(len(runner.plan(self.rows,a,'cv')),4);self.assertEqual(len(runner.plan(self.rows,a,'loso')),16)
    def test_original_output_is_protected(self):
        for dest in [self.old,self.old/'child',self.old.parent]:
            with self.assertRaisesRegex(ValueError,'separate'):runner.arguments(['--original-output',str(self.old),'--output',str(dest)])
    def test_matched_manifest_and_changed_hash_rejected(self):
        a=self.args();rows,_,_=runner.load_matched(a);self.assertEqual(len(rows),512);p=self.old/'scan_manifest.csv';original=p.read_bytes();p.write_text(p.read_text(encoding='utf-8-sig').replace(rows[0]['geometry_sha256'],'bad',1),encoding='utf-8-sig')
        try:
            with self.assertRaisesRegex(ValueError,'mismatch'):runner.load_matched(a)
        finally:p.write_bytes(original)
    def test_shuffle_is_deterministic_preserves_statistics_and_input(self):
        rng=np.random.default_rng(9);s={'color':rng.random((100,3),dtype=np.float32),'coord':rng.random((100,3),dtype=np.float32)};original=s['color'].copy();d=fm.alter_surface(s,'rgb_shuffled','scan')
        np.testing.assert_array_equal(original,s['color']);self.assertFalse(np.array_equal(original,d['color']));np.testing.assert_allclose(fm.colour_statistics_numpy(original),fm.colour_statistics_numpy(d['color']),atol=1e-6);np.testing.assert_array_equal(s['coord'],d['coord']);np.testing.assert_array_equal(d['color'],fm.alter_surface(s,'rgb_shuffled','scan')['color'])
    def test_gray_geometry_and_matched_width_settings(self):
        s={'color':np.eye(3,dtype=np.float32)};gray=fm.alter_surface(s,'gray_combined','s')['color'];np.testing.assert_allclose(gray[:,0],[.299,.587,.114]);np.testing.assert_array_equal(gray[:,0],gray[:,2]);np.testing.assert_array_equal(fm.alter_surface(s,'geometry_matched','s')['color'],0)
        jobs=[j for j in runner.plan(self.rows,self.args(),'colour') if j['seed']==42 and j['model']=='pointnext_s'];settings=[{k:v for k,v in j['surface'].items() if k!='pathway'} for j in jobs];self.assertEqual(len(jobs),6);self.assertTrue(all(s==settings[0] and s['inputs']=='geometry_rgb' for s in settings))
    def test_existing_prediction_reproduction_by_scan_id(self):
        rows=[r for r in self.rows if r['split']=='val'];prob=np.tile([.2,.3,.5],(64,1));p=self.root/'pred.csv';write_csv(p,list(reversed(predictions(rows,prob))));self.assertEqual(runner.compare_saved(rows,prob,p,.0005),0)
        prob[0]=[.6,.3,.1]
        with self.assertRaisesRegex(ValueError,'differs'):runner.compare_saved(rows,prob,p,.0005)
    def test_identity_and_original_files_unchanged(self):
        a=self.args();rows,audit,_=runner.load_matched(a);before={p.name:digest(p) for p in self.old.iterdir()};runner.save_identity(a,rows,audit);self.assertEqual(before,{p.name:digest(p) for p in self.old.iterdir()});a.folds=4
        with self.assertRaisesRegex(ValueError,'different'):runner.save_identity(a,rows,audit)
    def test_partial_and_complete_oof_report(self):
        a=self.args();a.output=self.root/'report';rows,audit,_=runner.load_matched(a);runner.save_identity(a,rows,audit);jobs=[j for j in runner.plan(rows,a,'cv') if j['model']=='pointnext_s' and j['variant']=='stats_only']
        for i,job in enumerate(jobs):
            dest=a.output/'runs'/job['name'];rr=[r for r in job['rows'] if r['split']=='assessment'];prob=np.tile([.2,.3,.5],(len(rr),1));write_json(dest/'config.json',{k:v for k,v in job.items() if k not in ['rows','name']});write_json(dest/'status.json',{'status':'complete'});write_json(dest/'metrics_assessment.json',score(rr,prob));write_csv(dest/'predictions_assessment.csv',predictions(rr,prob));rebuild(a.output)
            with (a.output/'followup_out_of_fold.csv').open(encoding='utf-8-sig') as stream:summary=list(csv.DictReader(stream))
            self.assertEqual(summary[0]['complete'],'True' if i==4 else 'False')
        self.assertEqual(summary[0]['scans'],'448')
    def test_core_runner_and_backend_hashes_preserved(self):
        root=Path(__file__).parent
        self.assertEqual(digest(root/'biocal3d_train_3d_investigations.py'),'9fde487d059ea6083fdbc29bc4ecd3bd51de02ef3fd775e1ac09f1270b1c9b98');self.assertEqual(digest(root/'official_backends.py'),'25ffb1743c4b3ee95ac659b6085cae5ba79d253d80d8ce06dbc1fb4a16e8d270')

@unittest.skipUnless(importlib.util.find_spec('torch'),'Torch unavailable here; run on training machine')
class TorchRoutingTests(unittest.TestCase):
    def test_pathway_information_and_gradient_routing(self):
        import torch
        fm.core.T=torch
        class Encoder(torch.nn.Module):
            def __init__(self):super().__init__();self.channels=4;self.layer=torch.nn.Linear(9,4)
            def forward(self,d):return self.layer(torch.cat([d['coord'],d['normal'],d['color']],-1)),d['coord']
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.encoder=Encoder();self.frozen=False;self.head=torch.nn.Linear(14,3)
            def forward(self,d,seed):return self.head(self.embedding(d,seed))
        original=fm.ORIGINAL_BUILD;fm.ORIGINAL_BUILD=lambda *args:(Model(),{});d={k:torch.rand(1,32,3) for k in ['coord','normal','color']}
        try:
            for pathway in fm.PATHWAYS:
                model,_=fm.build_model({'surface':{'pathway':pathway}},None,torch.device('cpu'));emb=model.embedding(d,123);self.assertEqual(tuple(emb.shape),(1,14))
                if pathway=='stats_only':
                    np.testing.assert_array_equal(emb.detach().numpy()[0,:8],0);model(d,123).sum().backward();self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
                if pathway in ['geometry_matched','rgb_encoder_only']:np.testing.assert_array_equal(emb.detach().numpy()[0,-6:],0)
                if pathway=='rgb_combined':np.testing.assert_allclose(emb.detach().numpy()[0,-6:],fm.colour_statistics_numpy(d['color'][0].numpy()),atol=1e-6)
        finally:fm.ORIGINAL_BUILD=original

if __name__=='__main__':unittest.main()
