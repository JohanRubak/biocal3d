"""Focused follow-up of BioCal3D's completed 23-run 3D investigation."""
from pathlib import Path
import argparse,copy,json,sys,traceback,gc
import numpy as np
from study_data import *
from study_report import metadata_table,baseline_controls,inspect_gallery,export_evaluation
import biocal3d_train_3d_investigations as core
import followup_models as fm

VERSION='3d-followup-1.0.0'
ORIGINAL_OUTPUT=Path(r'C:\Users\au662213\repos\biocal3d\3D branch\BioCal3D_3D_investigations_v1\BioCal3D_3D_investigations\runs\3d_prepared_20261013')
DEFAULT_OUTPUT=Path('runs/3d_followup_20261013')
FOCUS_SAMPLES=['BCG2T0-1','BCG4T2-5','BCG4T0-5','BCG4T1-2','BCG6T0-7','BCG4T1-5','BCG5T0-8']

def arguments(argv=None):
    raw=sys.argv[1:] if argv is None else list(argv)
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--action',choices=['audit','preflight','diagnostics','colour','cv','loso','report'],default='audit')
    p.add_argument('--original-output',type=Path,default=ORIGINAL_OUTPUT)
    p.add_argument('--mature-only',action='store_true')
    p.add_argument('--skip-controls',action='store_true')
    p.add_argument('--include-concerto-stats',action='store_true',help='Add matched Concerto statistics-only head to CV/LOSO; default already includes PointNeXt statistics-only.')
    own,remaining=p.parse_known_args(raw);a=core.arguments(remaining)
    for k,v in vars(own).items():setattr(a,k,v)
    if not any(x=='--output' or x.startswith('--output=') for x in remaining):a.output=DEFAULT_OUTPUT
    if a.final_evaluate:raise ValueError('This follow-up never opens reserved test for evaluation.')
    if a.mature_only and a.action=='diagnostics':raise ValueError('Existing RGB diagnostics reproduce the original full-cohort models. Run without --mature-only.')
    if a.mature_only and a.output==DEFAULT_OUTPUT:a.output=Path('runs/3d_followup_mature_20261013')
    a.output=a.output.expanduser().resolve();a.original_output=a.original_output.expanduser().resolve()
    if a.output==a.original_output or a.output.is_relative_to(a.original_output) or a.original_output.is_relative_to(a.output):raise ValueError('--output must be separate from the original results folder.')
    if len(set(a.seeds))!=len(a.seeds):raise ValueError('Repeated seeds are not allowed.')
    a.resume=True
    return a

def load_matched(a):
    raw,audit,excluded=load_snapshot(a.data_root,a.allow_incomplete)
    root=a.original_output;split=root/'reference_split.json';manifest=root/'scan_manifest.csv'
    if not split.is_file() or not manifest.is_file():raise FileNotFoundError('Original split/manifest not found. Pass --original-output "...\\runs\\3d_prepared_20261013".')
    rows=reference_split(raw,explicit=split)
    import csv
    with manifest.open(encoding='utf-8-sig',newline='') as f:old=list(csv.DictReader(f))
    previous={r['row_id']:r for r in old}
    if len(previous)!=len(old) or set(previous)!={r['row_id'] for r in rows}:raise ValueError('Original and current scan cohorts differ.')
    for r in rows:
        for key in ['physical_id','scanner','sample_name','stage','geometry_sha256','split']:
            if previous[r['row_id']].get(key)!=r[key]:raise ValueError(f'Original/current mismatch: {r["scan_id"]} {key}')
    return rows,audit,excluded

def base_surface(a):
    return dict(points=a.points,scale_mm=a.scale_mm,sampling_seed=a.sampling_seed,sampling='area',inputs='geometry_rgb',
                align_plane=False,xyz_only=False,surface_grid_mm=0.,grid_size=a.grid_size)

def plan(rows,a,action=None):
    action=action or a.action
    dev=[r for r in rows if r['split']!='test' and (not a.mature_only or r['group']>=3)]
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
    jobs=[]
    for protocol,tag,rr in partitions:
        assert_partition(rr)
        if any(r['split']=='test' for r in rr):raise AssertionError('Reserved test entered follow-up.')
        for model,pathway in configs:
            if model not in a.models:continue
            for seed in (a.seeds if protocol=='reference' else a.seeds[:1]):
                surface=dict(base_surface(a),pathway=pathway,followup_implementation=fm.implementation_hash())
                if pathway=='legacy_geometry':surface['inputs']='geometry'
                mode='frozen_pretrained' if model in core.FROZEN else ('frozen_random' if pathway=='stats_only' else 'scratch')
                jobs.append(dict(name=f'{protocol}_{tag}__{model}__{pathway}__seed{seed}',protocol=protocol,tag=tag,rows=rr,model=model,seed=seed,variant=pathway,surface=surface,mode=mode))
    return jobs

def save_identity(a,rows,audit):
    a.output.mkdir(parents=True,exist_ok=True)
    identity=dict(version=VERSION,index_sha256=audit['index_sha256'],original_split_sha256=digest(a.original_output/'reference_split.json'),
        cohort=[(r['row_id'],r['geometry_sha256'],r['split']) for r in rows],mature_only=a.mature_only,
        sampling={k:getattr(a,k) for k in ['points','scale_mm','sampling_seed','grid_size','neighbors']},folds=a.folds,split_seed=a.split_seed,implementation=fm.implementation_hash())
    identity['signature']=signature(identity)
    p=a.output/'followup_identity.json'
    if p.exists() and json.loads(p.read_text())['signature']!=identity['signature']:raise ValueError('Follow-up output belongs to different data/split/implementation/settings. Use a new --output.')
    write_json(p,identity);write_csv(a.output/'scan_manifest.csv',rows);write_json(a.output/'cohort_audit.json',audit)
    write_json(a.output/'reference_split.json',{'physical_ids':{part:sorted({r['physical_id'] for r in rows if r['split']==part}) for part in ['train','val','test']}})
    old=a.original_output/'comparison.csv'
    if old.is_file():(a.output/'original_comparison.csv').write_bytes(old.read_bytes())

def metadata(a,rows):
    import csv
    selected=[r for r in rows if r['split']!='test' and (not a.mature_only or r['group']>=3)]
    file=a.original_output/'geometry_metadata.csv'
    if file.exists():
        with file.open(encoding='utf-8-sig',newline='') as f:old={r['row_id']:r for r in csv.DictReader(f)}
        needed=['faces','vertices','log_faces','height_rq','height_ra','height_p05','height_p95','height_range','normal_spread','area_mm2','width_mm','length_mm']+[f'rgb_{stat}_{ch}' for stat in ['mean','sd'] for ch in 'rgb']
        if all(r['row_id'] in old and old[r['row_id']]['geometry_sha256']==r['geometry_sha256'] and all(c in old[r['row_id']] for c in needed) for r in selected):
            result=[]
            for row in selected:
                r=dict(row);r.update({k:float(old[row['row_id']][k]) for k in needed});result.append(r)
            write_csv(a.output/'geometry_metadata.csv',result);return result
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

def compare_saved(rows,prob,saved_path,tolerance):
    import csv
    with Path(saved_path).open(encoding='utf-8-sig',newline='') as f:saved=list(csv.DictReader(f))
    lookup={r['row_id']:r for r in saved}
    if len(lookup)!=len(saved) or set(lookup)!={r['row_id'] for r in rows}:raise ValueError('Saved prediction IDs do not match validation cohort.')
    other=np.array([[float(lookup[r['row_id']]['p_'+stage]) for stage in STAGES] for r in rows])
    if any(int(lookup[r['row_id']]['truth'])!=STAGES.index(r['stage']) for r in rows):raise ValueError('Saved validation truth mismatch.')
    difference=float(np.abs(prob-other).max())
    if difference>tolerance or not np.array_equal(prob.argmax(1),other.argmax(1)):raise ValueError(f'Existing checkpoint prediction differs from saved CSV (max {difference:.6g}). No diagnostic interpretation is allowed until provenance/settings agree.')
    return difference

def perturb(s,kind,row_id,seed):
    d={k:v.copy() for k,v in s.items()}
    if kind.startswith('rotate_'):
        angle=np.deg2rad(int(kind.split('_')[1]));c,t=np.cos(angle),np.sin(angle);rot=np.array([[c,-t,0],[t,c,0],[0,0,1]],np.float32)
        d['coord']=d['coord']@rot.T;d['normal']=d['normal']@rot.T;d['grid_origin']=d['coord'].min(0)
    elif kind=='shuffle_rgb':d['color']=d['color'][np.random.default_rng(seed_for('diagnostic-colour:'+row_id,seed)).permutation(len(d['color']))]
    elif kind=='gray':
        y=d['color']@np.array([.299,.587,.114],np.float32);d['color']=np.repeat(y[:,None],3,axis=1)
    elif kind=='point_order':
        ix=np.random.default_rng(seed_for('diagnostic-order:'+row_id,seed)).permutation(len(d['coord']))
        for k in ['coord','normal','color','tsgc','display_rgb','face_index','barycentric']:d[k]=d[k][ix]
    else:raise ValueError(kind)
    return d

def diagnostics(a,rows,device):
    import pandas as pd
    records=[];failed=[]
    for model_name in ['pointnext_s','concerto']:
        if model_name not in a.models:continue
        model=None
        try:
            name=f'reference_fixed__{model_name}__geometry_rgb__seed42';old=a.original_output/'runs'/name
            config=json.loads((old/'config.json').read_text());cfg=copy.copy(a)
            for key in ['lr','encoder_lr','batch_size','weight_decay','patience','head_dropout']:setattr(cfg,key,config[key])
            cfg.flash=bool(config['provenance'].get('flash',False))
            cfg.sampling_seed=config['surface']['sampling_seed']
            if model_name=='concerto':
                source_model=config['provenance']['model']
                if config['provenance'].get('checkpoint_sha256'):
                    if not cfg.concerto_checkpoint or digest(cfg.concerto_checkpoint)!=config['provenance']['checkpoint_sha256']:raise ValueError('Pass the exact original --concerto-checkpoint.')
                else:cfg.concerto_variant=source_model
            job={k:config[k] for k in ['protocol','tag','model','seed','variant','surface','mode']};job.update(name=name,rows=rows)
            expected={(r['row_id'],r['geometry_sha256'],r['split']) for r in config['cohort']}
            if expected!={(r['row_id'],r['geometry_sha256'],r['split']) for r in rows}:raise ValueError('Original checkpoint cohort differs.')
            model,provenance=fm.ORIGINAL_BUILD(job,cfg,device)
            for key in ['initial_state_sha256','adapter_runner_sha256','adapter_source_sha256']:
                if provenance[key]!=config['provenance'][key]:raise ValueError(f'Original model provenance mismatch: {key}')
            core.load_best(model,old/'best.pt',signature(config),device)
            val=[r for r in rows if r['split']=='val'];prob=core.predict(model,val,job,cfg,device)
            difference=compare_saved(val,prob,old/'predictions_val.csv',a.probability_tolerance)
            target=a.output/'existing_rgb_diagnostics'/model_name;target.mkdir(parents=True,exist_ok=True)
            export_evaluation(target,'val',val,prob,a.bootstrap)
            write_json(target/'provenance.json',dict(original_checkpoint_sha256=digest(old/'best.pt'),max_saved_probability_difference=difference,original_config_signature=signature(config),no_fitting=True))
            for kind in ['rotate_90','rotate_180','rotate_270','point_order','shuffle_rgb','gray']:
                pp=[]
                for row in val:
                    s,_=fm.ORIGINAL_SURFACE(row,job,cfg.output);s=perturb(s,kind,row['row_id'],a.sampling_seed)
                    with core.T.no_grad():p=model(core.tensor_data(s,device),int(s['seed'])).softmax(1)[0].cpu().numpy()
                    pp.append(p)
                pp=np.array(pp);metrics=export_evaluation(target,kind,val,pp,a.bootstrap)
                records.append(dict(model=model_name,perturbation=kind,changed_predictions=int((pp.argmax(1)!=prob.argmax(1)).sum()),mean_probability_difference=float(np.abs(pp-prob).mean()),**metrics))
            # Same fixed stage/scanner examples plus high-confidence RGB errors.
            selected=[]
            for stage in ['partial','baseline','clean']:
                for scanner in SCANNERS:
                    options=sorted([r for r in val if r['stage']==stage and r['scanner']==scanner],key=lambda r:r['sample_name'])
                    if options:selected.append(options[0])
            error_ix=[i for i,r in enumerate(val) if prob[i].argmax()!=STAGES.index(r['stage'])]
            error_ix.sort(key=lambda i:-float(prob[i].max()));selected.extend(val[i] for i in error_ix[:6])
            selected=list({r['row_id']:r for r in selected}.values());index={r['row_id']:i for i,r in enumerate(val)};audits=[]
            cfg.focus_examples=1
            for row in selected:
                core.focus_outputs(model,[row],prob[index[row['row_id']]][None],job,cfg,device,target,None)
                audits.extend(pd.read_csv(target/'focus'/'consistency_audit.csv').to_dict('records'))
                write_csv(target/'focus'/'all_examples_audit.csv',audits)
            write_csv(target/'focus'/'consistency_audit.csv',audits)
            write_json(target/'status.json',{'status':'complete','focus_examples':len(audits),'no_training':True})
            del model;model=None;gc.collect()
        except Exception as exc:traceback.print_exc();failed.append(dict(model=model_name,error=str(exc)))
        finally:
            if model is not None:del model
            gc.collect()
            if device.type=='cuda':core.T.cuda.empty_cache()
    write_csv(a.output/'existing_rgb_diagnostics'/'perturbations.csv',records);write_json(a.output/'diagnostics_failures.json',failed)
    if failed:raise RuntimeError('Some existing-checkpoint diagnostics failed; see diagnostics_failures.json. No original results were modified.')

def main(argv=None):
    a=arguments(argv)
    from followup_report import rebuild
    if a.action=='report':rebuild(a.output);return
    rows,audit,excluded=load_matched(a);save_identity(a,rows,audit);write_csv(a.output/'excluded_scans.csv',excluded)
    plans={action:plan(rows,a,action) for action in ['colour','cv','loso']}
    for action,jobs in plans.items():write_csv(a.output/f'plan_{action}.csv',[{k:v for k,v in j.items() if k!='rows'}|{p+'_scans':sum(r['split']==p for r in j['rows']) for p in ['train','val','assessment']} for j in jobs])
    print(f'Cohort: {len(rows)} scans; {len({r["physical_id"] for r in rows})} specimens; reserved test never evaluated.')
    print('Planned jobs: '+', '.join(f'{action}={len(jobs)}' for action,jobs in plans.items()),flush=True)
    rebuild(a.output)
    if a.plan:return
    if a.action=='audit':
        m=metadata(a,rows);geometry_audit(a,rows,m)
        if not a.skip_controls:baseline_controls([r for r in rows if r['split']!='test' and (not a.mature_only or r['group']>=3)],m,a.output/'controls'/'reference_followup',a.bootstrap)
        rebuild(a.output);return
    device=initialize_torch(a)
    if a.action=='diagnostics':diagnostics(a,rows,device);rebuild(a.output);return
    if a.action=='preflight':
        jobs=[j for j in plans['colour'] if j['seed']==a.seeds[0]]
        if 'sonata' in a.models:jobs.append(next(j for j in plans['cv'] if j['model']=='sonata'))
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
    m=None if a.skip_controls else metadata(a,rows);done_controls=set();failures=[]
    for i,job in enumerate(jobs,1):
        print(f'[{i}/{len(jobs)}] {job["name"]}',flush=True)
        try:
            control_key=job['protocol']+'_'+job['tag']
            if m is not None and control_key not in done_controls:
                baseline_controls(job['rows'],m,a.output/'controls'/control_key,a.bootstrap);done_controls.add(control_key)
            core.run_job(job,a,device)
            # Original runner limits focus export to old baseline; RGB focus is handled separately above.
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
    print('Completed. Open '+str(a.output/'followup_comparison.html'))

if __name__=='__main__':
    try:
        result=main();raise SystemExit(result or 0)
    except Exception as exc:print('ERROR: '+str(exc),file=sys.stderr);raise SystemExit(1)
