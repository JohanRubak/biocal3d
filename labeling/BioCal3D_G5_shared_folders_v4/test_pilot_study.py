"""Synthetic 512-scan integration checks. Run: python -m unittest -v test_pilot_study.py"""
import json,tempfile,unittest,contextlib,io
from pathlib import Path
import numpy as np
import vtk
from annotator.mesh_io import AnnotationStore,load_mesh,polydata
from prepare_pilot_study import prepare,write_csv
from analyse_labeling_agreement import analyse

class PilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);rows=[]
        self.launcher=self.root/'start_annotator.bat';self.launcher.write_text('@echo off\n')
        v=np.array([[0.,0,0],[1,0,0],[0,1,0],[1,1,0]]);f=np.array([[0,1,2],[1,3,2]])
        for group in range(1,7):
            for n in range(1,9):
                for t in range(2 if group<3 else 3):
                    for scanner in ('iTERO','LABscanner','TRIOS3','TRIOS5'):
                        sample=f'BCG{group}T{t}-{n}';folder=self.root/'samples'/scanner/f'G{group}'/sample;folder.mkdir(parents=True)
                        mesh=folder/'roi.vtp';w=vtk.vtkXMLPolyDataWriter();w.SetFileName(str(mesh));w.SetInputData(polydata(v,f,np.tile([120,45,90],(4,1))));w.Write()
                        rows.append(dict(scan_id=f'{scanner}__{sample}',physical_id=f'G{group}_S{n}',group=group,scanner=scanner,time_idx=t,
                                         stage='baseline' if t==0 else 'partial' if t==1 and group>=3 else 'clean',annotation_mesh=str(mesh.relative_to(self.root)),sample_dir=str(folder.relative_to(self.root))))
        self.rows=rows;write_csv(self.root/'annotation_inventory.csv',rows)
    def tearDown(self):self.tmp.cleanup()
    def run_prepare(self,**kwargs):
        kwargs={'calibration':4,'agreement':8,'agreement_g5':4,**kwargs}
        with contextlib.redirect_stdout(io.StringIO()):return prepare(self.root,app_launcher=self.launcher,**kwargs)
    def test_default_is_g5_only_two_calibration_six_agreement(self):
        with contextlib.redirect_stdout(io.StringIO()):assigned,c=prepare(self.root,app_launcher=self.launcher)
        self.assertEqual(c['counts']['agreement_bruna'],{'scans':72,'specimens':6})
        self.assertEqual(c['counts']['calibration_bruna'],{'scans':24,'specimens':2})
        self.assertEqual(sum(c['counts'][f'production_{r}']['scans'] for r in ('bruna','katrin')),416)
        self.assertTrue(all(pid.startswith('G5_') for pid in c['agreement_ids']+c['calibration_ids']))
        self.assertEqual(c['agreement_specimens_by_cohort'],{'G5':6,'standard_groups':0})
        self.assertEqual(c['sample_size_status'],'pilot_not_precision_certified')
        self.assertEqual(c['pilot_scope'],'G5_only')
        self.assertEqual(c['precision_target']['recommended_primary_cohort'],'G5')
        self.assertEqual(set(c['calibration_ids'])&set(c['agreement_ids']),set())
        self.assertEqual(len(set(c['calibration_ids']+c['agreement_ids'])),8)
        selected={pid:{} for pid in c['calibration_ids']+c['agreement_ids']}
        for row in assigned:
            if row['physical_id'] in selected:selected[row['physical_id']].setdefault(row['study_phase'],[]).append(row)
        self.assertTrue(all(len(phases)==1 and len(next(iter(phases.values())))==12 for phases in selected.values()))
        plan=self.root/'labeling_g5_pilot'
        self.assertEqual((plan/'agreement_bruna.csv').read_bytes(),(plan/'agreement_katrin.csv').read_bytes())
        for row in assigned:
            if row['study_phase']!='agreement':continue
            mesh=load_mesh(self.root/row['annotation_mesh'])
            for name in ('bruna','katrin'):
                AnnotationStore(self.root,self.root/'annotations',name).save(mesh,np.array([1,0],np.int8),True,'independent')
        with contextlib.redirect_stdout(io.StringIO()):summary=analyse(self.root,plan)
        self.assertEqual(summary['complete_pairs'],72)
        self.assertEqual(summary['partial_stage_primary']['scans'],24)
        self.assertEqual(summary['partial_stage_primary']['specimens'],6)
        self.assertEqual(set(summary['cohorts']),{'G5'})
        self.assertEqual(summary['raters'],['bruna','katrin'])
    def test_full_coverage_named_roles_partial_overlap_and_no_consensus(self):
        assigned,c=self.run_prepare();self.assertEqual(len(assigned),512);self.assertEqual(len({r['physical_id'] for r in assigned}),48)
        plan=self.root/'labeling_g5_pilot'
        self.assertEqual(c['counts']['calibration_bruna']['scans'],48);self.assertEqual(c['counts']['agreement_bruna']['scans'],96)
        self.assertEqual(set(c['calibration_ids'])&set(c['agreement_ids']),set())
        self.assertEqual(sum(pid.startswith('G5_') for pid in c['agreement_ids']),4)
        self.assertTrue(any(pid.startswith('G6_') for pid in c['agreement_ids']))
        self.assertEqual(c['counts']['production_bruna']['specimens']+c['counts']['production_katrin']['specimens'],36)
        self.assertEqual(c['raters'],['bruna','katrin']);self.assertFalse(c['metadata_blinded']);self.assertFalse((plan/'consensus.csv').exists())
        self.assertEqual(c['sample_size_status'],'pilot_not_precision_certified')
        self.assertEqual(c['precision_target']['half_width'],.05)
        self.assertEqual(c['eligible_agreement_specimens_after_calibration'],{'G5':4,'standard_groups':24,'all':28})
        self.assertEqual((plan/'agreement_bruna.csv').read_bytes(),(plan/'agreement_katrin.csv').read_bytes())
        def queue(name):
            import csv
            with (plan/(name+'.csv')).open(encoding='utf-8-sig',newline='') as src:return list(csv.DictReader(src))
        a=queue('all_bruna');b=queue('all_katrin');sa={r['annotation_mesh'] for r in a};sb={r['annotation_mesh'] for r in b}
        self.assertEqual(len(sa|sb),512);self.assertEqual(len(sa&sb),144)
        self.assertEqual(len(queue('production_bruna'))+len(queue('production_katrin')),368)
        owners={}
        for r in assigned:
            if r['study_phase']=='production':owners.setdefault(r['physical_id'],set()).add(r['assigned_to'])
        by_pid={}
        for r in assigned:by_pid.setdefault(r['physical_id'],set()).add(r['study_phase'])
        self.assertTrue(all(len(value)==1 for value in by_pid.values()))
        self.assertEqual(sum(value=={'calibration'} for value in by_pid.values()),4)
        self.assertEqual(sum(value=={'agreement'} for value in by_pid.values()),8)
        self.assertEqual({r['stage'] for r in queue('agreement_bruna')},{'baseline','partial','clean'})
        self.assertTrue(all(len(value)==1 for value in owners.values()))
        self.assertEqual(len(queue('experiment_johan')),512)
        self.assertTrue(all((self.root/'annotations'/name).is_dir() for name in ('bruna','katrin','johan')))
        self.assertIn('--annotator bruna --lock-annotator',(plan/'open_agreement_bruna.bat').read_text())
        self.assertIn('--annotator johan --lock-annotator',(plan/'open_experiment_johan.bat').read_text())
        with self.assertRaises(ValueError):self.run_prepare()
        other,c2=self.run_prepare(output=self.root/'repeat');self.assertEqual(assigned,other);self.assertEqual(c['agreement_ids'],c2['agreement_ids'])
        _,larger=self.run_prepare(output=self.root/'larger',agreement=10,agreement_g5=4)
        self.assertEqual(larger['counts']['agreement_bruna']['scans'],120)
        self.assertEqual(sum(pid.startswith('G5_') for pid in larger['agreement_ids']),4)
        self.assertEqual(sum(larger['counts'][f'production_{r}']['scans'] for r in ('bruna','katrin')),344)
        # Only Bruna/Katrin determine agreement; Johan can deliberately disagree.
        for row in queue('agreement_bruna'):
            mesh=load_mesh(self.root/row['annotation_mesh'])
            for name,labels in [('bruna',[1,0]),('katrin',[1,0]),('johan',[0,1])]:
                AnnotationStore(self.root,self.root/'annotations',name).save(mesh,np.array(labels,np.int8),True,'independent')
        with contextlib.redirect_stdout(io.StringIO()):summary=analyse(self.root,plan)
        self.assertEqual(summary['raters'],['bruna','katrin']);self.assertEqual(summary['complete_pairs'],96)
        self.assertEqual(summary['all_stages']['specimens'],8);self.assertEqual(summary['cohorts']['G5']['specimens'],4)
        self.assertEqual(summary['cohorts']['standard_groups']['specimens'],4)
        self.assertEqual(summary['partial_stage_primary']['scans'],32)
        self.assertEqual(summary['all_stages']['plaque_dice']['mean_of_specimen_means'],1)
    def test_explicit_ids_validation_and_insufficient_scanner_coverage(self):
        cal=['G5_S1','G5_S2','G5_S3','G5_S4'];agree=['G5_S5','G5_S6','G5_S7','G5_S8','G3_S1','G4_S1','G6_S1','G6_S2']
        _,c=self.run_prepare(calibration_ids=cal,agreement_ids=agree);self.assertEqual(c['calibration_ids'],cal);self.assertEqual(c['agreement_ids'],agree)
        with self.assertRaises(ValueError):self.run_prepare(output=self.root/'invalid',calibration_ids=['G5_S1','G5_S1'])
        self.assertFalse((self.root/'invalid').exists())
        rows=[r for r in self.rows if not (r['group']==5 and r['stage']=='partial' and r['scanner']=='iTERO')]
        write_csv(self.root/'annotation_inventory.csv',rows)
        with self.assertRaises(ValueError):self.run_prepare(output=self.root/'missing')
        self.assertFalse((self.root/'missing').exists())
        rows=[r for r in self.rows if not (r['group']==5 and r['stage']=='clean' and r['scanner']=='iTERO')]
        write_csv(self.root/'annotation_inventory.csv',rows)
        with self.assertRaises(ValueError):self.run_prepare(output=self.root/'missing_clean')
        self.assertFalse((self.root/'missing_clean').exists())

if __name__=='__main__':unittest.main()
