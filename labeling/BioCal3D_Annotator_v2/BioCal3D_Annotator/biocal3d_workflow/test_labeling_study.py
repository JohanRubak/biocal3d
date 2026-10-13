import csv,tempfile,unittest,random
from pathlib import Path
import numpy as np
import vtk
from annotator.mesh_io import polydata,AnnotationStore,load_mesh
from annotator.export_annotations import export_annotations
from labeling_study import create_plan,grouped_plan,seed_consensus,write_csv
from analyse_labeling_agreement import analyse,agreement_metrics

class StudyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.rows=[]
        v=np.array([[0.,0,0],[1,0,0],[0,1,0],[1,1,0]]);f=np.array([[0,1,2],[1,3,2]])
        for group in range(1,5):
            for specimen in range(1,4):
                for stage in range(2 if group<3 else 3):
                    for scanner in ('TRIOS3','LABscanner'):
                        sample=f'BCG{group}T{stage}-{specimen}';folder=self.root/'samples'/scanner/f'G{group}'/sample;folder.mkdir(parents=True)
                        p=folder/'roi.vtp';w=vtk.vtkXMLPolyDataWriter();w.SetFileName(str(p));w.SetInputData(polydata(v,f,np.tile([120,45,90],(4,1))));w.Write()
                        self.rows.append(dict(scan_id=scanner+'__'+sample,physical_id=f'G{group}_S{specimen}',group=group,scanner=scanner,stage=('baseline' if stage==0 else 'partial' if group>=3 and stage==1 else 'clean'),split='train',growth_condition='fresh' if group<3 else 'mature',annotation_mesh=str(p.relative_to(self.root)),sample_dir=str(folder.relative_to(self.root))))
        write_csv(self.root/'annotation_inventory.csv',self.rows)
    def tearDown(self):self.tmp.cleanup()
    def test_grouping_overlap_disjoint_production_and_reproducibility(self):
        rows=create_plan(self.root,calibration=2,agreement=4);by_pid={}
        for row in rows:
            pid=row['physical_id'];value=(row['study_phase'],row['assigned_to'])
            if pid in by_pid:self.assertEqual(by_pid[pid],value)
            by_pid[pid]=value
        self.assertEqual(sum(v[0]=='calibration' for v in by_pid.values()),2)
        self.assertEqual(sum(v[0]=='agreement' for v in by_pid.values()),4)
        self.assertEqual(sum(v[0]=='production' for v in by_pid.values()),6)
        shuffled=self.rows.copy();random.Random(77).shuffle(shuffled)
        other=grouped_plan(shuffled,2,4)
        self.assertEqual(by_pid,{r['physical_id']:(r['study_phase'],r['assigned_to']) for r in other})
        def ids(name):
            with (self.root/'labeling_plan'/(name+'.csv')).open(encoding='utf-8-sig') as s:return {r['annotation_mesh'] for r in csv.DictReader(s)}
        self.assertEqual(ids('agreement_dentist_1'),ids('agreement_dentist_2'))
        self.assertFalse(ids('production_dentist_1')&ids('production_dentist_2'))
        self.assertEqual(ids('all_dentist_1')|ids('all_dentist_2'),{r['annotation_mesh'] for r in self.rows})
        with self.assertRaises(ValueError):create_plan(self.root,calibration=2,agreement=4)
    def test_consensus_separate_and_agreement_exports(self):
        rows=create_plan(self.root,calibration=2,agreement=4);cases=[r for r in rows if r['study_phase']=='agreement'];plan=self.root/'labeling_plan'
        with self.assertRaises(ValueError):seed_consensus(self.root,plan)
        stores={n:AnnotationStore(self.root,self.root/'annotations',n) for n in ('dentist_1','dentist_2','consensus')};originals={}
        for row in cases:
            mesh=load_mesh(self.root/row['annotation_mesh'])
            for name in ('dentist_1','dentist_2'):
                stores[name].save(mesh,np.array([0,1],np.int8),True,'independent')
                p=stores[name].path_for(mesh.path);originals[p]=p.read_bytes()
        summary=analyse(self.root,plan)
        self.assertEqual(summary['complete_pairs'],len(cases));self.assertEqual(summary['all_stages']['plaque_dice']['mean_of_specimen_means'],1.)
        self.assertEqual(seed_consensus(self.root,plan),len(cases))
        for p,data in originals.items():self.assertEqual(p.read_bytes(),data)
        mesh=load_mesh(self.root/cases[0]['annotation_mesh']);labels,done,notes=stores['consensus'].load(mesh)
        self.assertFalse(done);np.testing.assert_array_equal(labels,[0,1]);self.assertIn('DRAFT',notes)
        self.assertEqual(seed_consensus(self.root,plan),0)
        result=export_annotations(self.root,self.root/'annotations',self.root/'exports')
        self.assertEqual(result['exported'],2*len(cases));self.assertEqual(result['skipped'],len(cases));self.assertEqual(result['failed'],0)
        with self.assertRaises(ValueError):export_annotations(self.root,self.root/'annotations',self.root/'annotations')
    def test_metrics_and_empty_masks(self):
        mesh=load_mesh(self.root/self.rows[0]['annotation_mesh'])
        result=agreement_metrics(mesh,np.array([0,1]),np.array([1,0]))
        self.assertEqual(result['plaque_dice'],0);self.assertEqual(result['area_weighted_kappa'],-1)
        result=agreement_metrics(mesh,np.array([0,0]),np.array([0,0]))
        self.assertTrue(np.isnan(result['plaque_dice']));self.assertTrue(result['both_empty_plaque']);self.assertEqual(result['area_agreement'],1)
        result=agreement_metrics(mesh,np.array([0,-1]),np.array([0,1]))
        self.assertEqual(result['joint_reviewed_fraction'],.5)

if __name__=='__main__':unittest.main()
