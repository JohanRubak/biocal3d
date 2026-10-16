"""Independent overlap agreement: area weighting, specimen means and cluster bootstrap."""
import argparse,csv,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from annotator.mesh_io import AnnotationStore,load_mesh
from labeling_study import resolve_root,write_csv,re_valid_plan_root


def agreement_metrics(mesh,a,b):
    tri=mesh.vertices[mesh.faces];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    active=mesh.active;valid=active&(a>=0)&(b>=0);total=float(area[active].sum());joint=float(area[valid].sum())
    confusion=np.array([[area[valid&(a==i)&(b==j)].sum() for j in (0,1)] for i in (0,1)],float)
    tn,fp,fn,tp=confusion.ravel();den=2*tp+fp+fn;union=tp+fp+fn
    observed=(tn+tp)/joint if joint else np.nan
    expected=float(np.dot(confusion.sum(0),confusion.sum(1))/joint**2) if joint else np.nan
    kappa=(observed-expected)/(1-expected) if joint and expected<1-1e-12 else np.nan
    result={'plaque_dice':2*tp/den if den else np.nan,'plaque_iou':tp/union if union else np.nan,
            'area_agreement':observed,'area_weighted_kappa':kappa,'joint_reviewed_fraction':joint/total if total else np.nan,
            'both_empty_plaque':bool(union==0),'joint_area':joint,'total_roi_area':total,
            'clean_clean_area':tn,'clean_plaque_area':fp,'plaque_clean_area':fn,'plaque_plaque_area':tp}
    for name,lab in [('rater_1',a),('rater_2',b)]:
        reviewed=active&(lab>=0);denominator=float(area[reviewed].sum())
        result[name+'_plaque_fraction']=float(area[active&(lab==1)].sum()/denominator) if denominator else np.nan
        result[name+'_reviewed_fraction']=denominator/total if total else np.nan
        result[name+'_uncertain_fraction']=float(area[active&(lab==-1)].sum()/total) if total else np.nan
        result[name+'_unreviewed_fraction']=float(area[active&(lab==-2)].sum()/total) if total else np.nan
    return result


def specimen_summary(rows,seed=42,iterations=2000):
    out={'scans':len(rows),'specimens':len({r['physical_id'] for r in rows})};rng=np.random.default_rng(seed)
    for metric in ('plaque_dice','plaque_iou','area_agreement','area_weighted_kappa','joint_reviewed_fraction'):
        groups=defaultdict(list)
        for row in rows:
            if np.isfinite(row[metric]):groups[row['physical_id']].append(row[metric])
        values=np.array([np.mean(groups[k]) for k in sorted(groups)])
        point=float(values.mean()) if len(values) else None;ci=None
        if len(values)>=2:
            boot=np.mean(values[rng.integers(0,len(values),(iterations,len(values)))],axis=1)
            ci=[float(x) for x in np.percentile(boot,[2.5,97.5])]
        out[metric]={'mean_of_specimen_means':point,'specimens_with_defined_metric':len(values),'ci95_specimen_bootstrap':ci}
    return out


def analyse(root,plan,output=None,annotations=None,rater1='dentist_1',rater2='dentist_2',seed=42):
    root=Path(root).resolve();plan=Path(plan)
    if not re_valid_plan_root(plan,root):raise ValueError('Plan does not match this prepared dataset.')
    if rater1==rater2:raise ValueError('Choose two distinct independent raters.')
    output=Path(output or plan/'agreement_results');output.mkdir(parents=True,exist_ok=True)
    stores=[AnnotationStore(root,annotations or root/'annotations',n) for n in (rater1,rater2)]
    with (plan/'agreement_dentist_1.csv').open(encoding='utf-8-sig',newline='') as stream:cases=list(csv.DictReader(stream))
    results=[];issues=[]
    for row in cases:
        try:
            if row.get('study_phase')!='agreement':raise ValueError('Only independent agreement cases belong in this analysis.')
            mesh=load_mesh(root/row['annotation_mesh']);a,adone,_=stores[0].load(mesh);b,bdone,_=stores[1].load(mesh)
            if not(adone and bdone):issues.append({'scan_id':row.get('scan_id'),'status':'INCOMPLETE','error':'Both ratings must be complete.'});continue
            results.append({k:row.get(k,'') for k in ('scan_id','physical_id','scanner','group','stage','split')}|agreement_metrics(mesh,a,b))
        except Exception as exc:issues.append({'scan_id':row.get('scan_id'),'status':'FAILED','error':str(exc)})
    write_csv(output/'agreement_by_scan.csv',results);write_csv(output/'agreement_missing_or_failed.csv',issues)
    summary={'expected_cases':len(cases),'complete_pairs':len(results),'missing_or_failed':len(issues),'raters':[rater1,rater2],
             'method':'Area-weighted binary agreement on jointly reviewed ROI faces. Empty/empty plaque Dice and IoU are undefined. Specimen means and physical-specimen bootstrap; calibration and consensus excluded.',
             'all_stages':specimen_summary(results,seed),'partial_stage_primary':specimen_summary([r for r in results if r['stage']=='partial'],seed)}
    scanners=[]
    for scanner in sorted({r['scanner'] for r in results}):
        subset=[r for r in results if r['scanner']==scanner];info=specimen_summary(subset,seed)
        scanners.append({'scanner':scanner,'scans':len(subset),'specimens':info['specimens'],**{key:value['mean_of_specimen_means'] for key,value in info.items() if isinstance(value,dict)}})
    write_csv(output/'agreement_by_scanner.csv',scanners)
    (output/'agreement_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    print(f'{len(results)}/{len(cases)} complete independent pairs; {len(issues)} incomplete/failed. Results: {output}')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path);p.add_argument('--plan',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--annotations',type=Path);p.add_argument('--rater-1',default='dentist_1');p.add_argument('--rater-2',default='dentist_2');p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    try:
        root=resolve_root(a.root);summary=analyse(root,a.plan or root/'labeling_plan',a.output,a.annotations,a.rater_1,a.rater_2,a.seed)
        return 2 if summary['missing_or_failed'] else 0
    except Exception as exc:print('ERROR:',exc);return 2

if __name__=='__main__':raise SystemExit(main())
