import csv
import json
import math
from pathlib import Path
import tempfile
import unittest
from plan_agreement_size import required_n, pilot_specimen_scores, planning_report


class PrecisionPlanningTests(unittest.TestCase):
    def test_known_precision_scenarios_and_capacity(self):
        report=planning_report(available=24)
        self.assertEqual([r['required_physical_specimens'] for r in report['scenarios']],[7,18,38])
        self.assertEqual([r['all_stage_scans_per_rater'] for r in report['scenarios']],[84,216,456])
        self.assertEqual([r['within_available_pool'] for r in report['scenarios']],[True,True,False])
        self.assertTrue(all(r['conditional_half_width_at_n']<=.05 for r in report['scenarios']))
        for bad in (0,-.1,float('nan')):
            with self.assertRaises(ValueError):required_n(bad)
        with self.assertRaises(ValueError):required_n(.1,half_width=0)

    def write_results(self,folder,rows):
        path=Path(folder)/'agreement_by_scan.csv'
        with path.open('w',newline='',encoding='utf-8-sig') as out:
            writer=csv.DictWriter(out,['physical_id','scanner','stage','plaque_dice','pilot_cohort'])
            writer.writeheader();writer.writerows(rows)
        return path

    def fixture(self):
        rows=[]
        for pid,cohort,base in [('A','standard_groups',.6),('B','standard_groups',.7),('C','standard_groups',.8),('D','G5',.1)]:
            for scanner,offset in zip(['iTERO','LABscanner','TRIOS3','TRIOS5'],[-.03,-.01,.01,.03]):
                for stage in ('baseline','partial','clean'):
                    rows.append(dict(physical_id=pid,scanner=scanner,stage=stage,plaque_dice=base+offset if stage=='partial' else 1,pilot_cohort=cohort))
        return rows

    def test_repeated_scans_do_not_inflate_n_or_mix_cohorts(self):
        with tempfile.TemporaryDirectory() as folder:
            path=self.write_results(folder,self.fixture())
            report=planning_report(from_results=path)
            self.assertEqual(report['pilot']['specimens'],3)
            self.assertEqual(set(report['pilot']['specimen_scores']),{'A','B','C'})
            self.assertAlmostEqual(report['pilot']['estimated_sd'],.1)
            self.assertEqual(report['scenarios'][0]['required_physical_specimens'],18)
            self.assertGreater(report['scenarios'][1]['required_physical_specimens'],18)
            scores,_=pilot_specimen_scores(path,cohort='all')
            self.assertEqual(len(scores),4)
            (Path(folder)/'plan.json').write_text(json.dumps({'precision_target':{'recommended_primary_cohort':'G5'},'eligible_agreement_specimens_after_calibration':{'G5':6,'standard_groups':24}}))
            planned=planning_report(plan=folder)
            self.assertEqual(planned['cohort'],'G5')
            self.assertEqual(planned['available_specimens'],6)

    def test_duplicate_and_undefined_scores_are_not_silent(self):
        with tempfile.TemporaryDirectory() as folder:
            rows=self.fixture()
            self.write_results(folder,rows+[rows[1]])
            with self.assertRaisesRegex(ValueError,'Duplicate'):pilot_specimen_scores(Path(folder)/'agreement_by_scan.csv')
            for r in rows:
                if r['physical_id']=='A' and r['stage']=='partial':r['plaque_dice']='nan'
            path=self.write_results(folder,rows)
            with self.assertRaisesRegex(ValueError,'found 2'):pilot_specimen_scores(path)
            for r in rows:
                if r['stage']=='partial':r['plaque_dice']=1
            path=self.write_results(folder,rows)
            with self.assertRaisesRegex(ValueError,'zero'):planning_report(from_results=path)

if __name__=='__main__':unittest.main()
