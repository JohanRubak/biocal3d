"""Follow-up report with matched pathways and explicitly incomplete OOF results."""
from pathlib import Path
import csv,html,json
from collections import defaultdict
import numpy as np
from study_data import STAGES,score,write_json,write_csv

LABELS={'geometry_matched':'Geometry only (matched width)','stats_only':'Global colour statistics only','rgb_encoder_only':'RGB encoder; no direct colour stats','rgb_combined':'RGB encoder + colour stats','rgb_shuffled':'Shuffled RGB + colour stats','gray_combined':'Grayscale encoder + intensity stats','legacy_geometry':'Original Sonata geometry baseline'}

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def table(rows,keys):
    def fmt(x):return f'{x:.4f}' if isinstance(x,float) else str(x if x is not None else '')
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(k.replace('_',' '))+'</th>' for k in keys)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(fmt(r.get(k)))+'</td>' for k in keys)+'</tr>' for r in rows)+'</tbody></table></div>'

def rebuild(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    identity=json.loads((out/'followup_identity.json').read_text()) if (out/'followup_identity.json').exists() else {}
    folds=identity.get('folds',5);mature=identity.get('mature_only',False)
    manifest=read_csv(out/'scan_manifest.csv') if (out/'scan_manifest.csv').exists() else []
    expected_ids={r['row_id'] for r in manifest if r['split']!='test' and (not mature or int(r['group'])>=3)}
    results=[];failures=[];preds={};configs={}
    for run in sorted((out/'runs').glob('*')) if (out/'runs').exists() else []:
        cfgfile=run/'config.json'
        if not cfgfile.exists():continue
        cfg=json.loads(cfgfile.read_text());configs[run.name]=cfg
        status=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
        if status.get('status')=='failed':failures.append(dict(run=run.name,error=status.get('error')))
        if status.get('status') not in ['complete','evaluated_checkpoint']:continue
        for part in ['val','assessment']:
            p=run/f'metrics_{part}.json'
            if not p.exists():continue
            m=json.loads(p.read_text());results.append(dict(run=run.name,protocol=cfg['protocol'],model=cfg['model'],variant=cfg['variant'],seed=cfg['seed'],partition=part,status=status.get('status'),
                balanced_accuracy=m['balanced_accuracy'],specimen_balanced_accuracy=m['specimen_balanced_accuracy'],partial_recall=m['recall'][1],macro_f1=m['macro_f1'],accuracy=m['accuracy'],scans=m['scans'],specimens=m['specimens'],best_epoch=m.get('best_epoch')))
            if (run/f'predictions_{part}.csv').exists():preds[run.name,part]=read_csv(run/f'predictions_{part}.csv')
    write_csv(out/'followup_comparison.csv',results)
    summary=[];reference=[r for r in results if r['protocol']=='reference' and r['partition']=='val' and not configs[r['run']]['tag'].startswith('interior')]
    for model in sorted({r['model'] for r in reference}):
        for variant in LABELS:
            rr=[r for r in reference if r['model']==model and r['variant']==variant]
            if rr:summary.append(dict(model=model,pathway=variant,seeds=len(rr),BA_mean=float(np.mean([r['specimen_balanced_accuracy'] for r in rr])),BA_seed_SD=float(np.std([r['specimen_balanced_accuracy'] for r in rr],ddof=1)) if len(rr)>1 else None,partial_recall_mean=float(np.mean([r['partial_recall'] for r in rr]))))
    write_csv(out/'colour_pathway_summary.csv',summary)
    deltas=[]
    for base in [r for r in reference if r['variant']=='rgb_combined']:
        for variant in ['geometry_matched','stats_only','rgb_encoder_only','rgb_shuffled','gray_combined']:
            rr=[r for r in reference if r['model']==base['model'] and r['seed']==base['seed'] and r['variant']==variant]
            if rr:deltas.append(dict(model=base['model'],seed=base['seed'],comparison='rgb_combined minus '+variant,delta_specimen_BA=base['specimen_balanced_accuracy']-rr[0]['specimen_balanced_accuracy']))
    write_csv(out/'colour_pathway_deltas.csv',deltas)
    groups=defaultdict(list)
    for (run,part),pp in preds.items():
        if part=='assessment':
            cfg=configs[run];groups[cfg['protocol'],cfg['model'],cfg['variant'],cfg['seed']].append((run,pp))
    pooled=[]
    for (protocol,model,variant,seed),entries in sorted(groups.items()):
        pp=[r for _,records in entries for r in records];ids=[r['row_id'] for r in pp]
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate OOF scan within '+str((protocol,model,variant,seed)))
        rows=[dict(physical_id=r['physical_id'],group=int(r['group']),stage=STAGES[int(r['truth'])]) for r in pp]
        prob=np.array([[float(r['p_'+stage]) for stage in STAGES] for r in pp]);m=score(rows,prob)
        expected_jobs=folds*(4 if protocol=='loso' else 1)
        pooled.append(dict(protocol=protocol,model=model,pathway=variant,seed=seed,completed_jobs=len(entries),expected_jobs=expected_jobs,
            complete=len(entries)==expected_jobs and set(ids)==expected_ids,scans=len(ids),specimens=m['specimens'],specimen_balanced_accuracy=m['specimen_balanced_accuracy'],partial_recall=m['recall'][1],macro_f1=m['macro_f1']))
    write_csv(out/'followup_out_of_fold.csv',pooled)
    # Pool the cheap controls with the same safeguards as learned models.
    cgroups=defaultdict(list)
    for folder in sorted((out/'controls').glob('*')) if (out/'controls').exists() else []:
        if not folder.name.startswith(('cv_','loso_')):continue
        protocol=folder.name.split('_',1)[0]
        for control in folder.iterdir():
            pred=control/'predictions_assessment.csv'
            if control.is_dir() and pred.exists():cgroups[protocol,control.name].append(read_csv(pred))
    pooled_controls=[]
    for (protocol,name),entries in sorted(cgroups.items()):
        pp=[r for entry in entries for r in entry];ids=[r['row_id'] for r in pp]
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate OOF control prediction: '+name)
        rows=[dict(physical_id=r['physical_id'],group=int(r['group']),stage=STAGES[int(r['truth'])]) for r in pp]
        prob=np.array([[float(r['p_'+stage]) for stage in STAGES] for r in pp]);m=score(rows,prob);n=folds*(4 if protocol=='loso' else 1)
        pooled_controls.append(dict(protocol=protocol,control=name,complete=len(entries)==n and set(ids)==expected_ids,completed_jobs=len(entries),expected_jobs=n,scans=len(ids),specimen_balanced_accuracy=m['specimen_balanced_accuracy'],partial_recall=m['recall'][1]))
    write_csv(out/'control_out_of_fold.csv',pooled_controls)
    if summary:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        models=sorted({r['model'] for r in summary});fig,axs=plt.subplots(1,len(models),figsize=(6.5*len(models),5),squeeze=False)
        for ax,model in zip(axs[0],models):
            rr=[r for r in summary if r['model']==model];ax.barh([LABELS[r['pathway']] for r in rr],[r['BA_mean'] for r in rr],xerr=[r['BA_seed_SD'] or 0 for r in rr],capsize=3,color=['#c66a43' if r['pathway']=='rgb_combined' else '#497a94' for r in rr]);ax.set_xlim(0,1.05);ax.invert_yaxis();ax.set_title(model);ax.set_xlabel('Validation specimen-balanced accuracy; mean ± seed SD')
        fig.tight_layout();fig.savefig(out/'colour_pathways.png',dpi=145);plt.close(fig)
    page='''<!doctype html><html><head><meta charset="utf-8"><title>BioCal3D targeted 3D follow-up</title><style>body{font:15px/1.55 system-ui;color:#203a4e;max-width:1300px;margin:30px auto;padding:20px}h1{font-size:29px}h2{margin-top:32px}table{border-collapse:collapse;font-size:12px}th,td{padding:7px;border-bottom:1px solid #dde5eb;text-align:left}.scroll{overflow:auto}img{max-width:100%}.note{background:#fff4da;padding:16px}summary{padding:10px;background:#edf3f6;cursor:pointer}</style></head><body><h1>BioCal3D · targeted 3D follow-up</h1>
<p class="note">Acquisition-stage classification, not biofilm segmentation. One grouped physical-specimen split of the approved cohort. Reserved test excluded. Colour pathway heads share encoder input width and head architecture within model/seed; absent pathways are zeroed. Geometry_matched is a new matched control, not a reuse of the old six-channel PointNeXt baseline.</p>'''
    if mature:page+='<p><strong>Mature-only experiment: G3–G6 only; fresh specimens excluded from fitting and assessment.</strong></p>'
    page+='<h2>Colour pathways</h2>'
    if (out/'colour_pathways.png').exists():page+='<img src="colour_pathways.png">'
    page+=table(summary,['model','pathway','seeds','BA_mean','BA_seed_SD','partial_recall_mean'])
    page+='<p>Higher performance after adding RGB does not by itself establish that geometry or local plaque boundaries are used. Compare combined with matched statistics-only, encoder-only, shuffled and grayscale pathways. Seed SD is not a confidence interval.</p>'
    page+='<h2>Specimen/scanner out-of-fold assessment</h2><p>Validation within each outer fold selects checkpoints. Assessment is never used for fitting or checkpoint selection. LOSO also excludes assessment subjects at every training scanner. Partial folds are explicitly incomplete. Each scan appears once per model/pathway/protocol/seed summary.</p>'+table(pooled,['protocol','model','pathway','seed','complete','completed_jobs','expected_jobs','scans','specimen_balanced_accuracy','partial_recall'])
    page+='<h2>Matching nuisance controls</h2>'+table(pooled_controls,['protocol','control','complete','completed_jobs','expected_jobs','scans','specimen_balanced_accuracy','partial_recall'])
    for f in sorted((out/'controls').glob('*/comparison.csv')) if (out/'controls').exists() else []:page+=f'<p><a href="{f.relative_to(out).as_posix()}">{html.escape(f.parent.name)} control results</a></p>'
    page+='<h2>Geometry QC</h2><p>Review raw exports and acquisition metadata before changing labels or excluding scans. No QC-based deletion is performed. The interior-ROI retraining comparison is linked from comparison.html. Voxel averaging does not restore missing scanner detail or remesh the original surface.</p>'
    qc=out/'qc'/'summary.json'
    if qc.exists():page+='<pre>'+html.escape(qc.read_text())+'</pre>'
    for p in sorted((out/'qc'/'sample_inspection').glob('*.png')):page+=f'<details><summary>{html.escape(p.stem)}</summary><img loading="lazy" src="{p.relative_to(out).as_posix()}"></details>'
    page+='<h2>Run metrics</h2>'+table(results,['run','partition','status','scans','specimens','specimen_balanced_accuracy','partial_recall','best_epoch'])
    page+='<h2>Failures</h2>'+table(failures,['run','error'])
    for p in sorted(out.glob('*failures*.json')):page+='<pre>'+html.escape(p.name+'\n'+p.read_text())+'</pre>'
    page+='<p>CSV: <a href="followup_comparison.csv">All runs</a> · <a href="colour_pathway_summary.csv">Colour summary</a> · <a href="colour_pathway_deltas.csv">Matched deltas</a> · <a href="followup_out_of_fold.csv">OOF models</a> · <a href="control_out_of_fold.csv">OOF controls</a> · <a href="qc/roughness_outliers.csv">QC outliers</a>.</p></body></html>'
    (out/'followup_comparison.html').write_text(page,encoding='utf-8')
