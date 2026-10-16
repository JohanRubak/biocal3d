"""Incremental reports and independently fitted nuisance / roughness controls."""
from pathlib import Path
import base64, html, json
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from study_data import STAGES,SCANNERS,geometry,geometric_features,color_features,score,cluster_interval,write_json,write_csv

def predictions(rows,prob):
    return [dict(row_id=r['row_id'],physical_id=r['physical_id'],group=r['group'],scanner=r['scanner'],sample_name=r['sample_name'],
                 truth=STAGES.index(r['stage']),prediction=int(p.argmax()),**{f'p_{c}':float(p[i]) for i,c in enumerate(STAGES)}) for r,p in zip(rows,prob)]

def export_evaluation(folder,part,rows,prob,bootstrap=300,plots=True):
    folder=Path(folder);m=score(rows,prob);write_json(folder/f'metrics_{part}.json',m)
    predrows=predictions(rows,prob);write_csv(folder/f'predictions_{part}.csv',predrows)
    errorrows=[dict(r,confidence=max(r[f'p_{c}'] for c in STAGES)) for r in predrows if r['truth']!=r['prediction']]
    write_csv(folder/f'errors_{part}.csv',sorted(errorrows,key=lambda r:-r['confidence']))
    write_json(folder/f'ci_{part}.json',cluster_interval(rows,prob,bootstrap))
    out=[]
    for scanner in sorted({r['scanner'] for r in rows}):
        ii=[i for i,r in enumerate(rows) if r['scanner']==scanner]
        out.append(dict(scanner=scanner,**score([rows[i] for i in ii],prob[ii])))
    write_json(folder/f'per_scanner_{part}.json',out);write_csv(folder/f'per_scanner_{part}.csv',out)
    if plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(1,2,figsize=(9,3.8));cm=np.array(m['confusion_matrix']);axs[0].imshow(cm,cmap='Blues')
        for i in range(3):
            for j in range(3):axs[0].text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()*.6 else 'black')
        axs[0].set_xticks(range(3),STAGES);axs[0].set_yticks(range(3),STAGES);axs[0].set_xlabel('Predicted');axs[0].set_ylabel('True');axs[0].set_title(part+' confusion matrix')
        axs[1].bar(STAGES,m['recall'],color=['#245b87','#c98424','#36836d']);axs[1].set_ylim(0,1);axs[1].set_title('Class recall')
        fig.tight_layout();fig.savefig(folder/f'diagnostics_{part}.png',dpi=140);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7.5,3.4));x=np.arange(len(out))
        ax.bar(x-.17,[r['accuracy'] for r in out],width=.3,label='Accuracy',color='#a6bdce')
        ax.bar(x+.17,[r['balanced_accuracy'] if r['balanced_accuracy'] is not None else np.nan for r in out],width=.3,label='Balanced accuracy',color='#245b87')
        ax.set_xticks(x,[r['scanner']+f'\n{r["scans"]} scans' for r in out]);ax.set_ylim(0,1);ax.legend();ax.set_title(part+' scanner performance')
        fig.tight_layout();fig.savefig(folder/f'scanners_{part}.png',dpi=140);plt.close(fig)
    return m

def metadata_table(rows,out):
    out=Path(out);data=[]
    for i,r in enumerate(rows):
        v,f,n,c,area,fn=geometry(r)
        data.append(dict(r,faces=len(f),vertices=len(v),log_faces=float(np.log(len(f))),has_rgb=bool(c is not None and c.max()>0),
                         **geometric_features(r),**(color_features(r) or {})))
        if i%32==0:print(f'Geometry/colour QC {i+1}/{len(rows)}',flush=True)
    write_csv(out/'geometry_metadata.csv',data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(10,7))
    for scanner,ax in zip(SCANNERS,axs.flat):
        values=[[r['faces'] for r in data if r['scanner']==scanner and r['stage']==c] for c in STAGES]
        ax.boxplot(values);ax.set_xticks([1,2,3],STAGES);ax.set_title(scanner);ax.set_ylabel('Mesh triangles');ax.set_yscale('log')
    fig.suptitle('Development cohort mesh density by scanner and stage · test excluded');fig.tight_layout();fig.savefig(out/'mesh_density.png',dpi=150);plt.close(fig)
    return data

def inspect_gallery(rows,samples,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from study_data import plane_frame
    folder=Path(output)/'sample_inspection';folder.mkdir(parents=True,exist_ok=True);notes=[]
    for sample in samples:
        chosen=[r for r in rows if r['sample_name'].upper()==sample.upper()]
        if not chosen:
            notes.append({'sample':sample,'status':'not in development cohort; may be reserved test or absent'});continue
        fig,axs=plt.subplots(2,len(SCANNERS),figsize=(13,6.2))
        for i,scanner in enumerate(SCANNERS):
            rr=[r for r in chosen if r['scanner']==scanner]
            if not rr:
                for ax in axs[:,i]:ax.text(.5,.5,'Missing scan',ha='center',transform=ax.transAxes);ax.axis('off')
                continue
            r=rr[0];v,f,n,c,a,fn=geometry(r);center,rot=plane_frame(v,f,a);flat=(v-center)@rot
            if c is not None and c.max()>0:
                axs[0,i].add_collection(PolyCollection(flat[f,:2],facecolors=c[f].mean(1),edgecolors='none',rasterized=True))
            else:axs[0,i].text(.5,.5,'RGB unavailable',ha='center',transform=axs[0,i].transAxes)
            height=flat[f,2].mean(1);pc=PolyCollection(flat[f,:2],array=height,cmap='coolwarm',edgecolors='none',rasterized=True)
            lim=max(float(np.max(np.abs(height))),.001);pc.set_clim(-lim,lim);axs[1,i].add_collection(pc);fig.colorbar(pc,ax=axs[1,i],fraction=.045,label='Height from fitted plane (mm)')
            axs[0,i].set_title(f'{scanner}\n{len(f):,} triangles; stage {r["stage"]}',fontsize=10)
            for ax in axs[:,i]:ax.set_xlim(flat[:,0].min(),flat[:,0].max());ax.set_ylim(flat[:,1].min(),flat[:,1].max());ax.set_aspect('equal');ax.axis('off')
        fig.suptitle(sample+' · true scanner colour and geometric height (individual colour scales)')
        fig.tight_layout();fig.savefig(folder/(sample+'.png'),dpi=150,bbox_inches='tight');plt.close(fig)
        notes.append({'sample':sample,'status':'rendered','scans':len(chosen)})
    write_json(folder/'status.json',notes)

def baseline_controls(rows,metadata,folder,bootstrap=300):
    # Each protocol trains its own preprocessing and classifier. Assessment and final
    # test subjects never fit the scaler, one-hot categories or regression weights.
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);byid={r['row_id']:r for r in metadata}
    selected=[dict(byid[r['row_id']],split=r['split']) for r in rows]
    train=[r for r in selected if r['split']=='train'];y=np.array([STAGES.index(r['stage']) for r in train])
    rough=['height_rq','height_ra','height_p05','height_p95','height_range','normal_spread','area_mm2','width_mm','length_mm']
    configs={'scanner_only':['scanner'],'face_count_only':['log_faces'],'scanner_face_count':['scanner','log_faces'],
             'roughness_geometry':rough,'colour_only':[f'rgb_{stat}_{ch}' for stat in ['mean','sd'] for ch in 'rgb']}
    output=[]
    for name,columns in configs.items():
        dest=folder/name;dest.mkdir(exist_ok=True)
        if any(any(c not in r for c in columns) for r in selected):
            write_json(dest/'status.json',{'status':'unavailable','reason':'Missing genuine vertex RGB; no zero-filled colour baseline.'});continue
        # pandas is only used here for named columns and official sklearn transformers.
        import pandas as pd
        trans=ColumnTransformer([(c,OneHotEncoder(handle_unknown='ignore') if c=='scanner' else StandardScaler(),[c]) for c in columns])
        model=make_pipeline(trans,LogisticRegression(C=1,class_weight='balanced',max_iter=3000))
        model.fit(pd.DataFrame(train)[columns],y)
        write_json(dest/'config.json',{'features':columns,'C':1,'class_weight':'balanced','fit_partition':'train','no_validation_tuning':True})
        for part in sorted({r['split'] for r in selected}-{'train'}):
            rr=[r for r in selected if r['split']==part];p=model.predict_proba(pd.DataFrame(rr)[columns]);m=export_evaluation(dest,part,rr,p,bootstrap,plots=False)
            output.append({'control':name,'partition':part,**m})
        write_json(dest/'status.json',{'status':'complete'})
    write_csv(folder/'comparison.csv',output)
    return output

def rebuild_report(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);rows=[];errors=[]
    for run in sorted((output/'runs').glob('*')) if (output/'runs').exists() else []:
        status=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
        if status.get('status')=='failed':errors.append({'run':run.name,'reason':status.get('error','')})
        if not (run/'config.json').exists():continue
        cfg=json.loads((run/'config.json').read_text())
        for file in sorted(run.glob('metrics_*.json')):
            part=file.stem.removeprefix('metrics_');m=json.loads(file.read_text())
            rows.append({'run':run.name,'protocol':cfg['protocol'],'variant':cfg['variant'],'model':cfg['model'],'seed':cfg['seed'],
                         'partition':part,'status':status.get('status',''),**m})
    write_csv(output/'comparison.csv',rows)
    write_csv(output/'failures.csv',errors)
    refs=[r for r in rows if r['protocol']=='reference' and r['partition']=='val' and r['variant']=='area_geometry']
    if refs:
        models=list(dict.fromkeys(r['model'] for r in refs));fig,axs=plt.subplots(1,2,figsize=(12,4.8))
        means=[];std=[];rec=[]
        for model in models:
            z=[r for r in refs if r['model']==model];means.append(np.mean([r['balanced_accuracy'] for r in z]));std.append(np.std([r['balanced_accuracy'] for r in z]));rec.append(np.mean([r['recall'] for r in z],axis=0))
        axs[0].barh(models,means,xerr=std,color='#245b87');axs[0].invert_yaxis();axs[0].set_xlim(0,1);axs[0].set_xlabel('Balanced accuracy (mean ± seed SD)');axs[0].set_title('Reference validation · area sampling')
        for i,val in enumerate(means):axs[0].text(min(val+.03,.9),i,f'{val:.1%}',va='center')
        axs[1].imshow(rec,vmin=0,vmax=1,cmap='Blues',aspect='auto');axs[1].set_xticks(range(3),STAGES);axs[1].set_yticks(range(len(models)),models);axs[1].set_title('Mean class recall')
        for i in range(len(models)):
            for j in range(3):axs[1].text(j,i,f'{rec[i][j]:.1%}',ha='center',va='center',color='white' if rec[i][j]>.6 else 'black')
        fig.tight_layout();fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)
    # OOF predictions: retain one record per scan and seed/protocol/variant/model.
    # Folds may be pooled; repeated seeds are summarised separately, never treated as new scans.
    pooled=[]
    for protocol in ['cv','loso']:
        for variant in sorted({r['variant'] for r in rows if r['protocol']==protocol}):
            for model in sorted({r['model'] for r in rows if r['protocol']==protocol and r['variant']==variant}):
                for seed in sorted({r['seed'] for r in rows if r['protocol']==protocol and r['variant']==variant and r['model']==model}):
                    selected=[r for r in rows if r['protocol']==protocol and r['variant']==variant and r['model']==model and r['seed']==seed and r['partition']=='assessment']
                    rr=[];pp=[];seen=set();duplicate=False
                    import csv
                    for x in selected:
                        with (output/'runs'/x['run']/'predictions_assessment.csv').open(encoding='utf-8-sig') as f:
                            for row in csv.DictReader(f):
                                if row['row_id'] in seen:duplicate=True;continue
                                seen.add(row['row_id']);rr.append({'row_id':row['row_id'],'physical_id':row['physical_id'],'group':int(row['group']),'stage':STAGES[int(row['truth'])]})
                                pp.append([float(row[f'p_{c}']) for c in STAGES])
                    if rr and not duplicate:pooled.append({'protocol':protocol,'variant':variant,'model':model,'seed':seed,'completed_assessment_jobs':len(selected),**score(rr,np.array(pp))})
    write_csv(output/'assessment_pooled.csv',pooled)
    def tab(records,keys):
        if not records:return '<p>No completed results yet.</p>'
        return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(f'{r.get(k):.4f}' if isinstance(r.get(k),float) else str(r.get(k,'')))+'</td>' for k in keys)+'</tr>' for r in records)+'</table>'
    page='<!doctype html><meta charset="utf-8"><title>BioCal3D 3D investigations</title><style>body{font:15px/1.5 system-ui;max-width:1300px;margin:30px auto;padding:20px}table{border-collapse:collapse;font-size:12px}td,th{padding:7px;border-bottom:1px solid #ddd}img{max-width:100%}.note{background:#eef4f9;padding:18px}</style><h1>BioCal3D · prepared-snapshot 3D investigations</h1>'
    page+='<p class="note">Reference validation selects checkpoints. CV and LOSO assessment subjects never select checkpoints. Global reserved test specimens are excluded from CV/LOSO. Baseline/frozen/fine-tuned runs are different adaptation strategies. Stage labels are not surface plaque annotations.</p>'
    for filename,label in [('cohort_audit.json','Cohort audit'),('experiment_plan.csv','Experiment plan'),('geometry_metadata.csv','Mesh and RGB QC'),('assessment_pooled.csv','Pooled assessment'),('failures.csv','Failures')]:
        if (output/filename).exists():page+=f'<a href="{filename}">{label}</a> · '
    if (output/'comparison.png').exists():page+='<p><img src="comparison.png"></p>'
    if (output/'mesh_density.png').exists():page+='<details><summary>Mesh density control</summary><img src="mesh_density.png"></details>'
    for f in sorted((output/'sample_inspection').glob('*.png')) if (output/'sample_inspection').exists() else []:
        page+=f'<details><summary>Inspect {html.escape(f.stem)} across scanners</summary><img src="{f.relative_to(output).as_posix()}"></details>'
    page+='<h2>Model results</h2>'+tab(rows,['run','partition','accuracy','balanced_accuracy','specimen_balanced_accuracy','partial_specimens','status'])
    page+='<h2>Out-of-fold assessment</h2><p>Incomplete suites remain partial. Each scan appears at most once within each protocol/model/variant/seed summary; seeds are not pooled as independent scans.</p>'+tab(pooled,['protocol','model','seed','completed_assessment_jobs','scans','balanced_accuracy'])
    page+='<h2>Nuisance and roughness controls</h2>'
    for base in sorted((output/'controls').glob('*/comparison.csv')) if (output/'controls').exists() else []:
        page+=f'<p><a href="{base.relative_to(output).as_posix()}">{html.escape(base.parent.name)}</a></p>'
    page+='<h2>Run details and focus figures</h2>'
    for run in sorted((output/'runs').glob('*')) if (output/'runs').exists() else []:
        page+=f'<details><summary>{html.escape(run.name)}</summary>'
        for f in sorted(run.glob('*.png'))+sorted((run/'focus').glob('*.png')):
            page+=f'<p><a href="{f.relative_to(output).as_posix()}">{html.escape(f.name)}</a><br><img loading="lazy" src="{f.relative_to(output).as_posix()}"></p>'
        page+='</details>'
    page+='<h2>Failed jobs</h2>'+tab(errors,['run','reason'])
    (output/'comparison.html').write_text(page,encoding='utf-8')

def learning_plot(history,dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(10,3.6));epoch=[r['epoch'] for r in history]
    for key,label in [('train_loss','Train'),('val_loss','Validation')]:axs[0].plot(epoch,[r[key] for r in history],label=label)
    axs[0].legend();axs[0].set_ylabel('Loss')
    axs[1].plot(epoch,[r['val_specimen_balanced_accuracy'] for r in history]);axs[1].set_ylabel('Validation specimen-weighted BA');axs[1].set_ylim(0,1)
    for ax in axs:ax.set_xlabel('Epoch')
    fig.tight_layout();fig.savefig(dest,dpi=140);plt.close(fig)
