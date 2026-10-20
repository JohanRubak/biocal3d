"""Merged BioCal3D 3D investigations on rebuilt, approved ROI snapshots."""
from pathlib import Path
import argparse,copy,json,sys,traceback,gc
import numpy as np
from study_data import *
from study_report import metadata_table,baseline_controls,inspect_gallery,export_evaluation
import biocal3d_train_3d_investigations as core
import followup_models as fm
from approved_data import default_root, load_approved
from extensions import rebuild_extensions, output_lock

VERSION='approved-investigations-2.0.0'
DEFAULT_OUTPUT=Path(__file__).resolve().parent/'runs'/'approved_rois_v2'
FOCUS_SAMPLES=['BCG2T0-1','BCG4T2-5','BCG4T0-5','BCG4T1-2','BCG6T0-7','BCG4T1-5','BCG5T0-8']

def arguments(argv=None):
    raw=sys.argv[1:] if argv is None else list(argv)
    p=argparse.ArgumentParser(add_help=False,description=__doc__)
    p.add_argument('--action',choices=['audit','preflight','baseline','colour','interior','cv','loso','report'],default='audit')
    p.add_argument('--split-policy',choices=['fresh','source'],default='fresh')
    p.add_argument('--mature-only',action='store_true')
    p.add_argument('--skip-controls',action='store_true')
    p.add_argument('--include-concerto-stats',action='store_true')
    p.add_argument('--interior-fraction',type=float,default=.8)
    if '--help' in raw or '-h' in raw:
        print(p.format_help())
        print('Default approved input: '+str(default_root()))
        print('Default output: '+str(DEFAULT_OUTPUT))
        print('Shared training options follow; legacy action flags are not used by this entry point.')
        core.arguments(['--help'])
    own,remaining=p.parse_known_args(raw);a=core.arguments(remaining)
    for k,v in vars(own).items():setattr(a,k,v)
    if not any(x=='--output' or x.startswith('--output=') for x in remaining):a.output=DEFAULT_OUTPUT
    if not any(x=='--data-root' or x.startswith('--data-root=') for x in remaining):a.data_root=default_root()
    if a.final_evaluate:raise ValueError('Reserved test stays locked in this development package.')
    if a.preflight or a.audit_only or a.controls_only or a.report_only:raise ValueError('Use --action preflight/audit/report instead of legacy action flags.')
    if a.allow_incomplete:raise ValueError('This approved-ROI study requires the complete rebuilt cohort.')
    if a.suite=='full':raise ValueError('Use --action cv and --action loso for targeted outer assessment.')
    if not 0<a.interior_fraction<1:raise ValueError('--interior-fraction must be between 0 and 1.')
    if a.mature_only and a.output==DEFAULT_OUTPUT:a.output=DEFAULT_OUTPUT.with_name('approved_rois_v2_mature')
    a.output=a.output.expanduser().resolve();a.data_root=a.data_root.expanduser().resolve()
    data_dir=a.data_root.parent if a.data_root.is_file() else a.data_root
    if a.output==data_dir or a.output.is_relative_to(data_dir) or data_dir.is_relative_to(a.output):raise ValueError('--output must be separate from the input snapshot.')
    if len(set(a.seeds))!=len(a.seeds):raise ValueError('Repeated seeds are not allowed.')
    a.resume=True
    return a

def load_matched(a):
    return load_approved(a)

def base_surface(a):
    return dict(points=a.points,scale_mm=a.scale_mm,sampling_seed=a.sampling_seed,sampling='area',inputs='geometry_rgb',
                align_plane=False,xyz_only=False,surface_grid_mm=0.,grid_size=a.grid_size)

def plan(rows,a,action=None):
    action=action or a.action
    dev=[r for r in rows if r['split']!='test' and (not a.mature_only or r['group']>=3)]
    if action=='baseline':
        cohort=[r for r in dev]
        return core.plan_jobs(cohort,a)
    if action in ['cv','loso']:
        partitions=[]
        # Restriction before splitting means mature-only folds contain mature subjects only.
        cohort=[r for r in rows if not a.mature_only or r['group']>=3]
        for k,rr in assessment_folds(cohort,a.folds,a.split_seed):
            if a.fold_indices is not None and k not in a.fold_indices:continue
            if action=='cv':partitions.append(('cv',f'fold{k}',rr))
            else:
                for scanner in SCANNERS:partitions.append(('loso',f'fold{k}_{scanner}',loso_rows(rr,scanner)))
        configs=[('pointnext_s','rgb_combined'),('pointnext_s','stats_only'),('concerto','rgb_combined'),('sonata','legacy_geometry')]
        if a.include_concerto_stats:configs.append(('concerto','stats_only'))
    else:
        partitions=[('reference','mature' if a.mature_only else 'followup',dev)]
        configs=[(model,pathway) for model in ['pointnext_s','concerto'] for pathway in fm.PATHWAYS]
    if action=='interior':configs=[(m,p) for m in ['pointnext_s','concerto'] for p in ['geometry_matched','stats_only','rgb_combined']]
    jobs=[]
    for protocol,tag,rr in partitions:
        assert_partition(rr)
        if any(r['split']=='test' for r in rr):raise AssertionError('Reserved test entered follow-up.')
        for model,pathway in configs:
            if model not in a.models:continue
            for seed in (a.seeds if protocol=='reference' else a.seeds[:1]):
                surface=dict(base_surface(a),pathway=pathway,followup_implementation=fm.implementation_hash())
                if action=='interior':surface['interior_fraction']=a.interior_fraction
                if pathway=='legacy_geometry':surface['inputs']='geometry'
                mode='frozen_pretrained' if model in core.FROZEN else ('frozen_random' if pathway=='stats_only' else 'scratch')
                if action=='interior':tag='interior_mature' if a.mature_only else 'interior'
                jobs.append(dict(name=f'{protocol}_{tag}__{model}__{pathway}__seed{seed}',protocol=protocol,tag=tag,rows=rr,model=model,seed=seed,variant=pathway,surface=surface,mode=mode))
    return jobs

def save_identity(a,rows,audit):
    a.output.mkdir(parents=True,exist_ok=True)
    identity=dict(version=VERSION,index_sha256=audit['index_sha256'],rebuild_sha256=audit['rebuild_sha256'],split_policy=audit['split_policy'],
        cohort=[(r['row_id'],r['geometry_sha256'],r['split']) for r in rows],mature_only=a.mature_only,
        sampling={k:getattr(a,k) for k in ['points','scale_mm','sampling_seed','grid_size','neighbors','surface_grid_mm','interior_fraction']},
        training={k:getattr(a,k) for k in ['lr','encoder_lr','batch_size','weight_decay','patience','head_dropout','epochs']},folds=a.folds,split_seed=a.split_seed,implementation=fm.implementation_hash())
    identity['signature']=signature(identity)
    p=a.output/'followup_identity.json'
    if p.exists() and json.loads(p.read_text())['signature']!=identity['signature']:raise ValueError('Follow-up output belongs to different data/split/implementation/settings. Use a new --output.')
    write_json(p,identity);write_csv(a.output/'scan_manifest.csv',rows);write_json(a.output/'cohort_audit.json',audit)
    write_json(a.output/'reference_split.json',{'physical_ids':{part:sorted({r['physical_id'] for r in rows if r['split']==part}) for part in ['train','val','test']}})


def metadata(a,rows):
    selected=[r for r in rows if r['split']!='test' and (not a.mature_only or r['group']>=3)]
    return metadata_table(selected,a.output)

def geometry_audit(a,rows,meta):
    import pandas as pd
    frame=pd.DataFrame(meta);wide=frame.pivot(index=['scanner','physical_id'],columns='stage',values='faces').reset_index()
    wide['baseline_clean_ratio']=wide['baseline']/wide['clean'];wide['baseline_more_faces']=wide['baseline']>wide['clean']
    write_csv(a.output/'qc'/'paired_face_counts.csv',wide.to_dict('records'))
    suspects=frame.sort_values('height_rq',ascending=False).head(16)
    write_csv(a.output/'qc'/'roughness_outliers.csv',suspects.to_dict('records'))
    write_json(a.output/'qc'/'summary.json',dict(pairs=len(wide),baseline_more_faces=int(wide.baseline_more_faces.sum()),
        median_face_ratio=wide.groupby('scanner').baseline_clean_ratio.median().to_dict(),
        interpretation='Descriptive correlated scanner/specimen pairs. Density-stage association does not establish biological topography. No automatic exclusion.'))
    selected=[r for r in rows if r['split']!='test' and (not a.mature_only or r['group']>=3)]
    from approved_data import interior_mask
    interior=[]
    for row in selected:
        v,f,vn,c,area,fn=geometry(row)
        if c is None or not c.max()>0:raise ValueError('Missing genuine vertex RGB: '+row['scan_id'])
        keep=interior_mask(row,v,f,a.interior_fraction)&(area>1e-12)
        if keep.sum()<17:raise ValueError('Insufficient interior triangles: '+row['scan_id'])
        interior.append(dict(scan_id=row['scan_id'],scanner=row['scanner'],sample_name=row['sample_name'],
            physical_id=row['physical_id'],split=row['split'],radius_fraction=a.interior_fraction,
            total_faces=len(f),interior_faces=int(keep.sum()),retained_area_fraction=float(area[keep].sum()/area.sum())))
    write_csv(a.output/'qc'/'interior_roi.csv',interior)
    inspect_gallery(selected,FOCUS_SAMPLES,a.output/'qc')

def initialize_torch(a):
    import torch
    core.T=torch
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else torch.device(a.device)
    if device.type=='cpu' and not a.allow_cpu:raise RuntimeError('CUDA unavailable. Activate biocal3d; --allow-cpu is only for a slow diagnostic.')
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('Requested CUDA is unavailable.')
    a.backends_root=core.resolve_backends(a.backends_root);fm.install()
    print(f'Device {device}; official backends {a.backends_root}',flush=True)
    return device

def main(argv=None):
    a=arguments(argv)
    with output_lock(a.output):
        return execute(a)


def execute(a):
    from followup_report import rebuild as base_report
    def rebuild(output):
        base_report(output);rebuild_extensions(output)
    if a.action=='report':rebuild(a.output);return
    rows,audit,excluded=load_matched(a);save_identity(a,rows,audit);write_csv(a.output/'excluded_scans.csv',excluded)
    plans={action:plan(rows,a,action) for action in ['baseline','colour','interior','cv','loso']}
    for action,jobs in plans.items():write_csv(a.output/f'plan_{action}.csv',[{k:v for k,v in j.items() if k!='rows'}|{p+'_scans':sum(r['split']==p for r in j['rows']) for p in ['train','val','assessment']} for j in jobs])
    partitions={}
    for jobs in plans.values():
        for job in jobs:partitions.setdefault(job['protocol']+'_'+job['tag'],job['rows'])
    for name,partition in partitions.items():write_csv(a.output/'partitions'/(name+'.csv'),partition)
    for part in ['train','val','test']:
        rr=[r for r in rows if r['split']==part]
        print(f'{part}: {len(rr)} scans; {len({r["physical_id"] for r in rr})} specimens')
    print(f'Cohort: {len(rows)} scans; {len({r["physical_id"] for r in rows})} specimens; reserved test never evaluated.')
    print('Planned jobs: '+', '.join(f'{action}={len(jobs)}' for action,jobs in plans.items()),flush=True)
    rebuild(a.output)
    if a.plan:return
    if a.action=='audit':
        m=metadata(a,rows);geometry_audit(a,rows,m)
        if not a.skip_controls:baseline_controls([r for r in rows if r['split']!='test' and (not a.mature_only or r['group']>=3)],m,a.output/'controls'/'reference_followup',a.bootstrap)
        rebuild(a.output);return
    device=initialize_torch(a)
    if a.action=='preflight':
        jobs=[j for j in plans['colour'] if j['seed']==a.seeds[0]]
        jobs += [j for j in plans['interior'] if j['seed']==a.seeds[0]]
        jobs += [j for j in plans['baseline'] if j['variant']=='area_geometry' and j['seed']==a.seeds[0]]
        if not jobs:raise ValueError('No model pathways selected for preflight.')
        results=[]
        for job in jobs:
            model,prov=core.build_model(job,a,device);row=next(r for r in job['rows'] if r['split']=='train');s,_=core.surface_data(row,job,a.output);model.eval()
            with core.T.no_grad():
                x=model(core.tensor_data(s,device),int(s['seed']));y=model(core.tensor_data(s,device),int(s['seed']))
            if not core.T.allclose(x,y,atol=a.probability_tolerance,rtol=0):raise ValueError('Preflight repeatability failed: '+job['name'])
            emb=model.embedding(core.tensor_data(s,device),int(s['seed'])).detach().cpu().numpy()[0]
            if job['variant']=='stats_only':
                np.testing.assert_array_equal(emb[:-6],0);np.testing.assert_allclose(emb[-6:],fm.colour_statistics_numpy(s['color']),atol=1e-5)
            if job['variant'] in ['geometry_matched','rgb_encoder_only']:np.testing.assert_array_equal(emb[-6:],0)
            results.append(dict(model=job['model'],pathway=job['variant'],status='passed',embedding_width=len(emb),provenance=prov));print('PASS '+job['name'],flush=True)
            del model;gc.collect()
            if device.type=='cuda':core.T.cuda.empty_cache()
        write_json(a.output/'preflight.json',results);rebuild(a.output);return
    jobs=plans.get(a.action)
    if jobs is None:raise ValueError('Unknown training action.')
    if not jobs:raise ValueError('No requested jobs. Check --models and --fold-indices.')
    write_csv(a.output/'experiment_plan.csv',[{k:v for k,v in j.items() if k!='rows'} for j in jobs])
    m=None if a.skip_controls or a.action=='interior' else metadata(a,rows);done_controls=set();failures=[]
    for i,job in enumerate(jobs,1):
        print(f'[{i}/{len(jobs)}] {job["name"]}',flush=True)
        try:
            control_key=job['protocol']+'_'+job['tag']
            if m is not None and control_key not in done_controls:
                baseline_controls(job['rows'],m,a.output/'controls'/control_key,a.bootstrap);done_controls.add(control_key)
            core.run_job(job,a,device)
            (a.output/'runs'/job['name']/'error.txt').unlink(missing_ok=True)
        except KeyboardInterrupt:
            rebuild(a.output);print('Interrupted. Rerun the same launcher to resume completed epochs.');return 130
        except Exception as exc:
            traceback.print_exc();failures.append(dict(job=job['name'],error=str(exc)));write_json(a.output/'runs'/job['name']/'status.json',{'status':'failed','error':str(exc)})
        finally:
            gc.collect()
            if device.type=='cuda':core.T.cuda.empty_cache()
            rebuild(a.output)
    write_json(a.output/f'failures_{a.action}.json',failures);rebuild(a.output)
    if failures:raise RuntimeError(f'{len(failures)} failed jobs. Correct the issue and rerun the same launcher; completed jobs are reused.')
    print('Completed. Open '+str(a.output/'comparison.html'))

if __name__=='__main__':
    try:
        result=main();raise SystemExit(result or 0)
    except Exception as exc:print('ERROR: '+str(exc),file=sys.stderr);raise SystemExit(1)
