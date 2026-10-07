"""Shared evaluation and specimen-clustered, paired model comparisons."""
from pathlib import Path
import csv, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from data3d import save_json, write_csv


def metrics(y,probs,classes):
    y=np.asarray(y,int); probs=np.asarray(probs); pred=probs.argmax(1)
    result=dict(n=len(y),accuracy=float(accuracy_score(y,pred)),
        balanced_accuracy=float(recall_score(y,pred,labels=list(range(len(classes))),average='macro',zero_division=0)),
        macro_f1=float(f1_score(y,pred,labels=list(range(len(classes))),average='macro',zero_division=0)),
        cross_entropy=float(-np.log(np.clip(probs[np.arange(len(y)),y],1e-12,1)).mean()))
    try:
        auc=roc_auc_score(y,probs[:,1]) if len(classes)==2 else roc_auc_score(y,probs,multi_class='ovr',average='macro',labels=list(range(len(classes))))
        result['roc_auc']=float(auc)
    except ValueError: result['roc_auc']=None
    return result

def read_predictions(path,classes):
    rows=list(csv.DictReader(open(path,encoding='utf-8-sig')))
    y=np.array([int(r['target']) for r in rows]); probs=np.array([[float(r['p_'+c]) for c in classes] for r in rows])
    return rows,y,probs

def plot_run(folder,classes):
    folder=Path(folder)
    history=json.loads((folder/'history.json').read_text())
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for part in ['train','val']:
        ax[0].plot([r['epoch'] for r in history],[r[part]['cross_entropy'] for r in history],label=part)
        ax[1].plot([r['epoch'] for r in history],[r[part]['balanced_accuracy'] for r in history],label=part)
    ax[0].set_ylabel('Cross-entropy (unweighted evaluation)'); ax[1].set_ylabel('Balanced accuracy'); ax[1].set_ylim(0,1)
    for a in ax: a.set_xlabel('Epoch'); a.legend(); a.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(folder/'learning_curves.png',dpi=180); plt.close(fig)
    rows,y,p=read_predictions(folder/'test_predictions.csv',classes)
    cm=confusion_matrix(y,p.argmax(1),labels=list(range(len(classes))))
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for a,m,title in zip(axes,[cm,cm/np.maximum(cm.sum(1,keepdims=True),1)],['Test counts','Recall by true class']):
        a.imshow(m,cmap='Blues'); a.set_xticks(range(len(classes)),classes,rotation=20); a.set_yticks(range(len(classes)),classes)
        a.set_xlabel('Predicted'); a.set_ylabel('True'); a.set_title(title)
        for i in range(len(classes)):
            for j in range(len(classes)): a.text(j,i,f'{m[i,j]:.2f}' if title.startswith('Recall') else str(m[i,j]),ha='center',va='center',color='black')
    fig.tight_layout(); fig.savefig(folder/'confusion_matrix.png',dpi=180); plt.close(fig)
    for key in ['scanner','growth_condition','group']:
        summary=[]
        for value in sorted({r[key] for r in rows}):
            ix=[i for i,r in enumerate(rows) if r[key]==value]
            ms=metrics(y[ix],p[ix],classes)
            # Missing-class slices report observed-class recall, separately from the fixed-class metric.
            present=sorted(set(y[ix]))
            ms['observed_class_balanced_accuracy']=float(recall_score(y[ix],p[ix].argmax(1),labels=present,average='macro',zero_division=0))
            ms['classes_present']=';'.join(classes[c] for c in present)
            summary.append({key:value,**ms})
        write_csv(folder/f'test_by_{key}.csv',summary)


def compare(output,bootstrap=1000):
    output=Path(output); dirs=sorted(p.parent for p in output.glob('*/test_predictions.csv'))
    if not dirs: raise ValueError('No test predictions found.')
    configs=[json.loads((p/'config.json').read_text()) for p in dirs]
    classes=configs[0]['classes']; split_hash=configs[0]['split_hash']; index_hash=configs[0]['index_hash']
    if any(c['classes']!=classes or c['split_hash']!=split_hash or c['index_hash']!=index_hash or c['targets_hash']!=configs[0]['targets_hash'] for c in configs):
        raise ValueError('Comparison requires identical classes, dataset snapshot and splits.')
    loaded=[read_predictions(p/'test_predictions.csv',classes) for p in dirs]
    ids=[r['scan_id'] for r in loaded[0][0]]
    if any([r['scan_id'] for r in x[0]]!=ids or not np.array_equal(x[1],loaded[0][1]) for x in loaded):
        raise ValueError('Test predictions must have identical ordered scan IDs and targets.')
    clusters=sorted({r['physical_id'] for r in loaded[0][0]})
    cluster_indices={c:np.array([i for i,r in enumerate(loaded[0][0]) if r['physical_id']==c]) for c in clusters}
    rng=np.random.default_rng(20261008)
    draws=[]
    for _ in range(bootstrap):
        ix=np.concatenate([cluster_indices[c] for c in rng.choice(clusters,len(clusters),replace=True)])
        if len(set(loaded[0][1][ix]))==len(classes): draws.append(ix)
    records=[]; boots=[]
    for folder,config,(rows,y,p) in zip(dirs,configs,loaded):
        stats=metrics(y,p,classes)
        values=np.array([[metrics(y[ix],p[ix],classes)[k] for k in ['balanced_accuracy','macro_f1']] for ix in draws]).reshape(-1,2)
        boots.append(values)
        rec=dict(run=folder.name,model=config['model'],input=config['input_mode'],seed=config['seed'],**stats)
        for j,k in enumerate(['balanced_accuracy','macro_f1']):
            lo,hi=np.quantile(values[:,j],[.025,.975]) if len(values) else (None,None)
            rec[k+'_ci_low']=None if lo is None else float(lo); rec[k+'_ci_high']=None if hi is None else float(hi)
        rec['bootstrap_valid']=len(draws); records.append(rec)
    write_csv(output/'comparison.csv',records); save_json(output/'comparison.json',records)
    differences=[]
    for a in range(len(dirs)):
        for b in range(a+1,len(dirs)):
            for j,k in enumerate(['balanced_accuracy','macro_f1']):
                vals=boots[a][:,j]-boots[b][:,j]
                lo,hi=np.quantile(vals,[.025,.975]) if len(vals) else (None,None)
                differences.append(dict(run_a=dirs[a].name,run_b=dirs[b].name,metric=k,
                    difference_a_minus_b=records[a][k]-records[b][k],
                    ci_low=None if lo is None else float(lo),ci_high=None if hi is None else float(hi),clusters=len(clusters)))
    write_csv(output/'paired_differences.csv',differences)
    fig,axes=plt.subplots(1,2,figsize=(max(10,len(records)*1.8),5))
    labels=[('TSGCNet' if r['model']=='tsgcnet' else 'PointNeXt-S')+'\n'+('XYZ + normals + RGB' if r['input']=='geometry_rgb' else 'XYZ + normals')+'\nSeed '+str(r['seed']) for r in records]
    for a,k in zip(axes,['balanced_accuracy','macro_f1']):
        v=np.array([r[k] for r in records]); a.bar(range(len(records)),v,color=['#4169a1' if r['model']=='tsgcnet' else '#cf772f' for r in records])
        for i,r in enumerate(records):
            if r[k+'_ci_low'] is not None: a.vlines(i,r[k+'_ci_low'],r[k+'_ci_high'],color='black',linewidth=2)
        a.set_xticks(range(len(records)),labels,rotation=25,ha='right'); a.set_ylim(0,1); a.set_ylabel(k.replace('_',' ')); a.grid(axis='y',alpha=.2)
    fig.suptitle(f'Test set: {len(ids)} scans / {len(clusters)} physical specimens; 95% specimen bootstrap intervals')
    fig.tight_layout(); fig.savefig(output/'comparison.png',dpi=180); plt.close(fig)
    import html
    table='<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in records[0])+'</tr>'
    for r in records: table+='<tr>'+''.join('<td>'+html.escape(f'{v:.3f}' if isinstance(v,float) else str(v))+'</td>' for v in r.values())+'</tr>'
    links=''.join(f'<h3>{html.escape(p.name)}</h3><a href="{p.name}/test_predictions.csv">Predictions</a> · <a href="{p.name}/test_by_scanner.csv">Scanner results</a><br><img width="750" src="{p.name}/confusion_matrix.png"><img width="750" src="{p.name}/learning_curves.png">' for p in dirs)
    page='<!doctype html><meta charset="utf-8"><title>BioCal3D comparison</title><style>body{font:16px system-ui;margin:30px}td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}img{max-width:100%}</style><h1>BioCal3D classifier comparison</h1><p>Stage classification, shared specimen splits. Intervals resample whole physical specimens. These are exploratory estimates from a small test cohort; repeated seeds do not add independent specimens.</p><img src="comparison.png" width="1200">'+table+'</table>'+links
    (output/'comparison.html').write_text(page,encoding='utf-8')
    return records
