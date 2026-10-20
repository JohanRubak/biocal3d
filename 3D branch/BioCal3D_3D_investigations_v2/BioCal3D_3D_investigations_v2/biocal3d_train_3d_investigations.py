#!/usr/bin/env python3
"""Train the official four BioCal3D 3D candidates on a prepared snapshot.
Run --help / --audit-only / --plan before training. See README.md.
"""
from __future__ import annotations
import argparse, contextlib, copy, gc, hashlib, json, math, os, random, sys, time, traceback
from pathlib import Path
import numpy as np
from study_data import *
from study_report import metadata_table,baseline_controls,export_evaluation,predictions,rebuild_report,learning_plot,inspect_gallery

MODELS=['pointnext_s','tsgcnet','sonata','concerto']
FROZEN=['sonata','concerto']
T=None
SURFACE_CODE_SHA=digest(Path(__file__).with_name('study_data.py'))

def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data-root',type=Path,default=Path(DEFAULT_ROOT),help='Prepared folder containing dataset_index.json. Never a raw-data search root.')
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'runs'/'approved_rois_v2')
    p.add_argument('--backends-root',type=Path,help='Existing third_party containing PointNeXt, TSGCNet, sonata and concerto.')
    p.add_argument('--models',nargs='+',choices=MODELS,default=MODELS)
    p.add_argument('--ablation-models',nargs='+',choices=MODELS,default=['pointnext_s','concerto'])
    p.add_argument('--suite',choices=['core','investigations','full'],default='investigations')
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--split-seed',type=int,default=42)
    p.add_argument('--split-json',type=Path,help='Optional physical_ids train/val/test JSON; otherwise preserve complete snapshot split or create a grouped split.')
    p.add_argument('--sampling-seed',type=int,default=123)
    p.add_argument('--folds',type=int,default=5)
    p.add_argument('--fold-indices',type=int,nargs='+',help='Optional subset, e.g. --fold-indices 0 for a short CV/LOSO pilot.')
    p.add_argument('--points',type=int,default=2048)
    p.add_argument('--scale-mm',type=float,default=10.)
    p.add_argument('--grid-size',type=float,default=.01,help='Pretrained voxel size in scaled coordinates, .01 × 10 mm = .1 mm.')
    p.add_argument('--surface-grid-mm',type=float,default=.15,help='Point-cloud voxel averaging control; does not restore missing mesh detail.')
    p.add_argument('--neighbors',type=int,default=16)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--patience',type=int,default=25)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--encoder-lr',type=float,default=.00001)
    p.add_argument('--weight-decay',type=float,default=.0001)
    p.add_argument('--head-dropout',type=float,default=.25)
    p.add_argument('--finetune',action='store_true',help='Add a Concerto pretrained fine-tuning job (low encoder LR, eval-mode encoder, batch size 1).')
    p.add_argument('--focus-examples',type=int,default=12)
    p.add_argument('--focus-errors',type=int,default=6,help='Additional high-confidence errors per explained reference model.')
    p.add_argument('--inspect-samples',nargs='*',default=['BCG3T1-6'],help='Cross-scanner RGB/height inspection, development subjects only.')
    p.add_argument('--focus-grid',type=int,default=4)
    p.add_argument('--probability-tolerance',type=float,default=.0005)
    p.add_argument('--bootstrap',type=int,default=300)
    p.add_argument('--device',default='auto')
    p.add_argument('--allow-cpu',action='store_true')
    p.add_argument('--allow-incomplete',action='store_true',help='Proceed with a reviewed incomplete cohort; never silently drop records.')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--live',action='store_true',help='Refresh training curves and comparison.html after each epoch.')
    p.add_argument('--audit-only',action='store_true',help='Validate snapshot and export development-set mesh/RGB QC; no torch/backends needed.')
    p.add_argument('--plan',action='store_true',help='Write the exact cohort, partitions and experiment plan, then stop before opening meshes for training.')
    p.add_argument('--controls-only',action='store_true',help='Run nuisance, geometric roughness and colour-statistics baselines without torch.')
    p.add_argument('--preflight',action='store_true',help='Check each requested official backend and repeatability on one real training scan.')
    p.add_argument('--final-evaluate',action='store_true',help='Evaluate existing reference best checkpoints on reserved test subjects; no training. Requires --resume.')
    p.add_argument('--evaluate-only',action='store_true',help='Export validation/assessment, rigid checks and focus from existing best checkpoints without further training; requires --resume.')
    p.add_argument('--report-only',action='store_true')
    p.add_argument('--sonata-checkpoint',type=Path)
    p.add_argument('--concerto-checkpoint',type=Path)
    p.add_argument('--concerto-variant',default='concerto_small')
    p.add_argument('--flash',action='store_true')
    p.add_argument('--attention-patch',type=int,default=128)
    a=p.parse_args(argv)
    if min(a.points,a.scale_mm,a.grid_size,a.neighbors,a.epochs,a.patience,a.batch_size,a.lr,a.encoder_lr,a.folds,a.bootstrap,a.probability_tolerance)<=0:p.error('Counts, scales and learning rates must be positive.')
    if a.points<=a.neighbors+1 or a.focus_grid<2 or min(a.focus_examples,a.focus_errors)<0:p.error('Too few points or invalid focus settings.')
    if not 0<=a.head_dropout<1:p.error('head dropout must be in [0,1).')
    if a.fold_indices is not None and any(k<0 or k>=a.folds for k in a.fold_indices):p.error('Fold indices must be between zero and folds-1.')
    if a.final_evaluate and not a.resume:p.error('--final-evaluate requires --resume and saved best checkpoints.')
    if a.evaluate_only and not a.resume:p.error('--evaluate-only requires --resume and saved best checkpoints.')
    if a.final_evaluate and a.evaluate_only:p.error('Choose either --final-evaluate or --evaluate-only.')
    return a

def resolve_backends(value):
    if value:
        if not value.is_dir():raise FileNotFoundError(value)
        return value.resolve()
    here=Path(__file__).resolve().parent
    for parent in [here,*list(here.parents)[:5]]:
        for candidate in [parent/'third_party',parent/'biocal3d_3d_benchmark'/'third_party',parent/'biocal3d_3d_benchmark'/'biocal3d_3d_benchmark'/'third_party']:
            if candidate.is_dir() and any((candidate/x).exists() for x in ['PointNeXt','TSGCNet','sonata','concerto']):return candidate.resolve()
    raise FileNotFoundError('Official backend directory not found. Pass --backends-root "...\\biocal3d_3d_benchmark\\third_party". No substitute networks are used.')

def plan_jobs(reference,a):
    base={'points':a.points,'scale_mm':a.scale_mm,'sampling_seed':a.sampling_seed,'sampling':'area','inputs':'geometry',
          'align_plane':False,'xyz_only':False,'surface_grid_mm':0.,'grid_size':a.grid_size}
    jobs=[]
    def add(protocol,tag,rows,model,seed,variant,change=None,mode=None):
        if model not in a.models:return
        config=dict(base,**(change or {}));mode=mode or ('frozen_pretrained' if model in FROZEN else 'scratch')
        name=f'{protocol}_{tag}__{model}__{variant}__seed{seed}'
        jobs.append({'name':name,'protocol':protocol,'tag':tag,'rows':rows,'model':model,'seed':seed,'variant':variant,'surface':config,'mode':mode})
    for model in a.models:
        for seed in a.seeds:add('reference','fixed',reference,model,seed,'area_geometry')
    ablations=[m for m in a.ablation_models if m in a.models]
    if a.suite in ['investigations','full']:
        variants=[('face_centers',{'sampling':'face_centers'}),('xyz_only',{'xyz_only':True}),
                  ('voxel_matched',{'surface_grid_mm':a.surface_grid_mm}),('plane_aligned',{'align_plane':True}),
                  ('geometry_rgb',{'inputs':'geometry_rgb'})]
        for model in ablations:
            for name,change in variants:
                if model=='tsgcnet' and name=='voxel_matched':continue
                add('reference','fixed',reference,model,a.seeds[0],name,change)
        if 'pointnext_s' in a.models:add('reference','fixed',reference,'pointnext_s',a.seeds[0],'frozen_random_pointnext',mode='frozen_random')
    if a.finetune and 'concerto' in a.models:add('reference','fixed',reference,'concerto',a.seeds[0],'pretrained_finetune',mode='finetune')
    if a.suite=='full':
        for fold,rows in assessment_folds(reference,a.folds,a.split_seed):
            if a.fold_indices is not None and fold not in a.fold_indices:continue
            for model in ablations:
                add('cv',f'fold{fold}',rows,model,a.seeds[0],'area_geometry')
                for scanner in SCANNERS:add('loso',f'fold{fold}_{scanner}',loso_rows(rows,scanner),model,a.seeds[0],'area_geometry')
    if a.final_evaluate:jobs=[j for j in jobs if j['protocol']=='reference']
    return jobs

def set_seed(seed):
    random.seed(seed);np.random.seed(seed);T.manual_seed(seed)
    if T.cuda.is_available():T.cuda.manual_seed_all(seed)
    T.backends.cudnn.benchmark=False;T.backends.cudnn.deterministic=True

@contextlib.contextmanager
def local_rng(seed,device):
    # Official point-transformer serialisation can shuffle orders. Fix the row seed
    # for both full-surface inference and every occlusion; restore caller RNG state.
    py=random.getstate();npstate=np.random.get_state()
    devices=[device.index if device.index is not None else T.cuda.current_device()] if device.type=='cuda' else []
    try:
        with T.random.fork_rng(devices=devices):
            random.seed(seed);np.random.seed(seed);T.manual_seed(seed)
            if device.type=='cuda':T.cuda.manual_seed_all(seed)
            yield
    finally:random.setstate(py);np.random.set_state(npstate)

def tensor_data(surface,device):
    return {k:T.from_numpy(np.array(surface[k],copy=True)).float().unsqueeze(0).to(device) for k in ['coord','normal','color','tsgc','grid_origin']}

def state_hash(module):
    h=hashlib.sha256()
    for name,value in module.state_dict().items():h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def surface_data(row,job,output):
    config=job['surface'];key=signature([VERSION,SURFACE_CODE_SHA,row['geometry_sha256'],config,job['model']=='tsgcnet'])
    folder=output/'surface_cache'/key[:20];folder.mkdir(parents=True,exist_ok=True);dest=folder/(row['row_id']+'.npz')
    if dest.exists():
        with np.load(dest,allow_pickle=False) as z:
            if str(z['signature'].item())!=key:raise ValueError('Surface cache signature mismatch.')
            return {k:z[k].copy() for k in z.files if k!='signature'},key
    data=make_surface(row,config,job['model']);np.savez_compressed(dest,**data,signature=np.array(key))
    return data,key

def build_model(job,a,device):
    import official_backends as b
    set_seed(job['seed']);cfg=copy.copy(a);cfg.inputs=job['surface']['inputs'];cfg.grid_size=job['surface']['grid_size']
    if job['model']=='pointnext_s':encoder=b.PointNextEncoder(cfg)
    elif job['model']=='tsgcnet':encoder=b.TSGCEncoder(cfg)
    else:encoder=b.PretrainedEncoder(job['model'],cfg,device)
    encoder=encoder.to(device);frozen=job['mode'] in ['frozen_pretrained','frozen_random']
    if job['mode']=='finetune':
        for p in encoder.parameters():p.requires_grad=True
    if frozen:
        for p in encoder.parameters():p.requires_grad=False
    # Infer actual feature width from the installed backend, rather than assuming
    # a particular Sonata/Concerto model revision's channel count.
    probe_row=next(r for r in job['rows'] if r['split']=='train')
    probe,_=surface_data(probe_row,job,a.output);encoder.eval()
    with T.no_grad(),local_rng(int(probe['seed']),device):
        local,_=encoder(tensor_data(probe,device))
    encoder.channels=int(local.shape[-1])
    class Model(T.nn.Module):
        def __init__(self):
            super().__init__();self.encoder=encoder;self.is_pretrained=job['model'] in FROZEN;self.frozen=frozen
            channels=2*encoder.channels+(6 if cfg.inputs=='geometry_rgb' else 0)
            self.head=T.nn.Sequential(T.nn.LayerNorm(channels),T.nn.Linear(channels,128),T.nn.GELU(),T.nn.Dropout(a.head_dropout),T.nn.Linear(128,3))
        def train(self,mode=True):
            super().train(mode)
            # Preserve pretrained normalisation/dropout behaviour even during optional
            # fine-tuning; gradients can flow through an encoder in eval mode.
            if self.frozen or self.is_pretrained:self.encoder.eval()
            return self
        def embedding(self,d,seed):
            with local_rng(seed,device):
                local,_=self.encoder(d)
                value=T.cat([local.mean(1),local.amax(1)],1)
                if cfg.inputs=='geometry_rgb':value=T.cat([value,d['color'].mean(1)*2-1,(d['color']*2-1).amax(1)],1)
                return value
        def forward(self,d,seed):return self.head(self.embedding(d,seed))
    model=Model().to(device)
    return model,dict(encoder.provenance,initial_state_sha256=state_hash(encoder),mode=job['mode'],
                      adapter_runner_sha256=digest(__file__),adapter_source_sha256=digest(Path(__file__).with_name('official_backends.py')),
                      actual_feature_channels=encoder.channels)

def embedding_cache(model,rows,job,a,device,provenance):
    folder=a.output/'feature_cache'/signature([provenance,job['surface']])[:20];folder.mkdir(parents=True,exist_ok=True)
    values={};model.eval()
    for i,r in enumerate(rows):
        surface,key=surface_data(r,job,a.output);seed=int(surface['seed']);dest=folder/(r['row_id']+'.npz');sig=signature([key,provenance,seed])
        if dest.exists():
            with np.load(dest,allow_pickle=False) as z:
                if str(z['signature'].item())!=sig:raise ValueError('Feature cache provenance differs.')
                values[r['row_id']]=z['embedding'].copy()
        else:
            with T.no_grad():value=model.embedding(tensor_data(surface,device),seed)[0].cpu().numpy()
            if not np.isfinite(value).all():raise FloatingPointError('Nonfinite pretrained embedding.')
            np.savez_compressed(dest,embedding=value,signature=np.array(sig));values[r['row_id']]=value
        if i%32==0:print(f'{job["model"]}: features {i+1}/{len(rows)}',flush=True)
    return values

def predict(model,rows,job,a,device,features=None,perturb=None):
    model.eval();prob=[]
    with T.no_grad():
        for r in rows:
            if features is not None and perturb is None:logits=model.head(T.from_numpy(features[r['row_id']]).float()[None].to(device))
            else:
                s,_=surface_data(r,job,a.output)
                if perturb:s=transformed(s,perturb)
                logits=model(tensor_data(s,device),int(s['seed']))
            p=logits.softmax(1)[0].cpu().numpy()
            if not np.isfinite(p).all():raise FloatingPointError('Nonfinite predictions.')
            prob.append(p)
    return np.array(prob)

def train(model,rows,job,a,device,run,key,features):
    rr=[r for r in rows if r['split']=='train'];vv=[r for r in rows if r['split']=='val']
    ys=np.array([STAGES.index(r['stage']) for r in rr]);counts=np.bincount(ys,minlength=3)
    if (counts==0).any():raise ValueError('Training lacks a class.')
    weights=T.tensor(len(ys)/(3*counts),dtype=T.float32,device=device);criterion=T.nn.CrossEntropyLoss(weight=weights)
    groups=[{'params':model.head.parameters(),'lr':a.lr}]
    if not model.frozen:groups.append({'params':[p for p in model.encoder.parameters() if p.requires_grad],'lr':a.encoder_lr if job['mode']=='finetune' else a.lr})
    opt=T.optim.AdamW(groups,weight_decay=a.weight_decay)
    history=[];best=-math.inf;stale=0;epoch0=0
    last=run/'last.pt';bestpath=run/'best.pt'
    def checkpoint(epoch):
        # Frozen encoders are fingerprinted, not duplicated in every seed checkpoint.
        return {'version':VERSION,'signature':key,'epoch':epoch,'best':best,'stale':stale,'history':history,
                'head':model.head.state_dict(),'encoder':None if model.frozen else model.encoder.state_dict(),
                'optimizer':opt.state_dict(),'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':T.get_rng_state(),
                                                   'cuda':T.cuda.get_rng_state_all() if T.cuda.is_available() else None}}
    def save(path,data):
        temp=path.with_name(path.name+'.tmp');T.save(data,temp);temp.replace(path)
    if last.exists():
        if not a.resume:raise FileExistsError(f'{last} already exists. Use --resume or a new output.')
        c=T.load(last,map_location=device,weights_only=False)
        if c['signature']!=key:raise ValueError('Checkpoint settings/data/backend signature changed. Use a new output.')
        model.head.load_state_dict(c['head'])
        if c['encoder'] is not None:model.encoder.load_state_dict(c['encoder'])
        opt.load_state_dict(c['optimizer']);history=c['history'];epoch0=c['epoch'];best=c['best'];stale=c['stale']
        random.setstate(c['rng']['python']);np.random.set_state(c['rng']['numpy']);T.set_rng_state(c['rng']['torch'].cpu())
        if c['rng']['cuda'] is not None and T.cuda.is_available():T.cuda.set_rng_state_all([v.cpu() for v in c['rng']['cuda']])
    batch_size=1 if job['mode']=='finetune' else a.batch_size
    for epoch in range(epoch0+1,a.epochs+1):
        if stale>=a.patience:break
        start=time.perf_counter();model.train();order=np.random.permutation(len(rr));losses=[]
        for begin in range(0,len(order),batch_size):
            indices=order[begin:begin+batch_size]
            if len(indices)==1 and batch_size>1 and not model.frozen:continue
            opt.zero_grad(set_to_none=True)
            if features is not None:
                x=T.tensor(np.array([features[rr[i]['row_id']] for i in indices]),dtype=T.float32,device=device);logits=model.head(x)
            elif job['model'] in FROZEN:
                values=[]
                for i in indices:
                    s,_=surface_data(rr[i],job,a.output);values.append(model(tensor_data(s,device),int(s['seed'])))
                logits=T.cat(values,0)
            else:
                surfaces=[surface_data(rr[i],job,a.output)[0] for i in indices]
                data={k:T.tensor(np.stack([s[k] for s in surfaces]),device=device,dtype=T.float32) for k in ['coord','normal','color','tsgc']}
                # Scratch encoders do not shuffle transformer serialisation. One fixed
                # seed context encloses the batch, preserving head dropout outside it.
                logits=model.head(model.embedding(data,seed_for(f'train:{epoch}:{begin}',job['seed'])))
            labels=T.tensor(ys[indices],dtype=T.long,device=device);loss=criterion(logits,labels)
            if not T.isfinite(loss):raise FloatingPointError('Nonfinite training loss.')
            loss.backward();T.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();losses.append((float(loss.detach()),len(indices)))
        if not losses:raise ValueError('No training batch; reduce --batch-size.')
        probabilities=predict(model,vv,job,a,device,features);metrics=score(vv,probabilities);selected=metrics['specimen_balanced_accuracy']
        improved=selected>best+1e-8
        if improved:best=selected;stale=0
        else:stale+=1
        history.append({'epoch':epoch,'train_loss':sum(x*n for x,n in losses)/sum(n for x,n in losses),'val_loss':metrics['loss'],
                        'val_balanced_accuracy':metrics['balanced_accuracy'],'val_specimen_balanced_accuracy':selected,'seconds':time.perf_counter()-start})
        c=checkpoint(epoch);save(last,c)
        if improved:save(bestpath,c)
        write_csv(run/'history.csv',history);write_json(run/'status.json',{'status':'training','completed_epochs':epoch,'best_specimen_ba':best})
        print(f'{job["name"]} {epoch:03d}: train {history[-1]["train_loss"]:.4f}; val BA {metrics["balanced_accuracy"]:.3f}; specimen BA {selected:.3f}',flush=True)
        if a.live:learning_plot(history,run/'training.png');rebuild_report(a.output)
    if not bestpath.exists():raise RuntimeError('No completed epoch/checkpoint.')
    return load_best(model,bestpath,key,device),history

def load_best(model,path,key,device):
    if not path.exists():raise FileNotFoundError(f'No trained checkpoint: {path}')
    c=T.load(path,map_location=device,weights_only=False)
    if c['signature']!=key:raise ValueError('Best checkpoint signature differs from this dataset/experiment.')
    model.head.load_state_dict(c['head'])
    if c['encoder'] is not None:model.encoder.load_state_dict(c['encoder'])
    model.eval();return c['epoch']

def focus_outputs(model,rows,prob,job,a,device,run,features):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    selected=[]
    for stage in ['partial','baseline','clean']:
        for scanner in SCANNERS:
            candidates=[i for i,r in enumerate(rows) if r['stage']==stage and r['scanner']==scanner]
            if candidates:
                # Select identical scan IDs across models; inspect high-confidence
                # errors separately in errors_val.csv rather than changing examples.
                candidates.sort(key=lambda i:rows[i]['sample_name'])
                selected.append(candidates[0])
    dest=run/'focus';dest.mkdir(exist_ok=True);audit=[]
    selected=selected[:a.focus_examples]
    errors=[i for i,r in enumerate(rows) if int(prob[i].argmax())!=STAGES.index(r['stage'])]
    errors.sort(key=lambda i:-float(prob[i].max()))
    selected=list(dict.fromkeys(selected+errors[:a.focus_errors]))
    for index in selected:
        r=rows[index];s,_=surface_data(r,job,a.output)
        baseline=predict(model,[r],job,a,device,features=None)[0]
        difference=float(np.max(np.abs(baseline-prob[index])))
        if difference>a.probability_tolerance or baseline.argmax()!=prob[index].argmax():
            write_json(dest/'consistency_failure.json',{'row_id':r['row_id'],'max_probability_difference':difference,'saved':prob[index].tolist(),'direct':baseline.tolist()})
            raise ValueError(f'Focus rejected: direct prediction differs from saved probabilities by {difference:.6g}. Check serialisation / backend nondeterminism.')
        xy=s['coord'][:,:2];n=a.focus_grid;extent=np.maximum(np.ptp(xy,axis=0),1e-8)
        cells=np.floor((xy-xy.min(0))/extent*n).astype(int).clip(0,n-1);region=cells[:,0]+n*cells[:,1]
        target=int(prob[index].argmax());sensitivity=np.zeros(len(xy),np.float32);model.eval()
        with T.no_grad():
            for cell in np.unique(region):
                keep=np.flatnonzero(region!=cell)
                if len(keep)<a.neighbors+1:continue
                # Removal is refilled to fixed N. This is a controlled sensitivity
                # perturbation, not an anatomical segmentation or causal plaque map.
                rng=np.random.default_rng(seed_for(f'{r["row_id"]}:{cell}',a.sampling_seed));ix=rng.choice(keep,len(xy),replace=True)
                pert={k:(v[ix] if k in ['coord','normal','color','tsgc'] else v) for k,v in s.items()}
                p=model(tensor_data(pert,device),int(s['seed'])).softmax(1)[0,target].item()
                sensitivity[region==cell]=float(baseline[target])-p
        np.savez_compressed(dest/(r['row_id']+'.npz'),coord=s['coord'],sensitivity=sensitivity,face_index=s['face_index'],barycentric=s['barycentric'],
                            display_rgb=s['display_rgb'],explained_class=target,original_probabilities=baseline,saved_probabilities=prob[index],seed=s['seed'])
        fig,axs=plt.subplots(1,3,figsize=(12,4))
        v,f,vn,rgb,area,fn=geometry(r);v=((v-s['center_mm'])@s['rotation'])/job['surface']['scale_mm']
        has_rgb=rgb is not None and rgb.max()>0
        for j in [0,2]:
            if has_rgb:
                facecolors=rgb[f].mean(1);collection=PolyCollection(v[f,:2],facecolors=facecolors,edgecolors='none',rasterized=True);axs[j].add_collection(collection)
                axs[j].set_xlim(v[:,0].min(),v[:,0].max());axs[j].set_ylim(v[:,1].min(),v[:,1].max())
            else:axs[j].text(.5,.5,'True sample RGB unavailable',ha='center',va='center',transform=axs[j].transAxes)
        vmax=max(float(np.max(np.abs(sensitivity))),.01)
        im=axs[1].scatter(xy[:,0],xy[:,1],c=sensitivity,cmap='coolwarm',vmin=-vmax,vmax=vmax,s=5)
        if has_rgb:axs[2].scatter(xy[:,0],xy[:,1],c=sensitivity,cmap='coolwarm',vmin=-vmax,vmax=vmax,s=6,alpha=.5)
        for ax in axs:ax.set_aspect('equal');ax.axis('off')
        axs[0].set_title('Actual scanner RGB · mesh rendering');axs[1].set_title(f'{n}×{n} occlusion sensitivity');axs[2].set_title('RGB + sensitivity overlay')
        fig.colorbar(im,ax=axs[1],fraction=.04,label='Probability drop after removal')
        fig.suptitle(f'{r["scanner"]} {r["sample_name"]} | true {r["stage"]} | predicted {STAGES[target]} {prob[index,target]:.1%}')
        fig.tight_layout();fig.savefig(dest/(r['row_id']+'.png'),dpi=160,bbox_inches='tight');plt.close(fig)
        audit.append({'row_id':r['row_id'],'scanner':r['scanner'],'sample_name':r['sample_name'],'true_stage':r['stage'],
                      'explained_class':STAGES[target],'max_probability_difference':difference,'has_actual_rgb':bool(has_rgb),'regions':n*n})
    write_csv(dest/'consistency_audit.csv',audit)

def run_job(job,a,device):
    run=a.output/'runs'/job['name'];run.mkdir(parents=True,exist_ok=True)
    rows=[r for r in job['rows'] if r['split']!='test' or a.final_evaluate]
    if job['surface']['inputs']=='geometry_rgb':
        bad=[r['scan_id'] for r in rows if not check_surface_rgb(r)]
        if bad:raise ValueError(f'RGB experiment unavailable: {len(bad)} scans lack genuine RGB (first {bad[:3]}). Repair preparation; no zero-filled fallback.')
    model,provenance=build_model(job,a,device)
    config={k:v for k,v in job.items() if k not in ['rows','name']}
    config.update(provenance=provenance,lr=a.lr,encoder_lr=a.encoder_lr,batch_size=a.batch_size,weight_decay=a.weight_decay,
                  patience=a.patience,head_dropout=a.head_dropout,version=VERSION,torch_version=T.__version__,
                  cohort=[{k:r[k] for k in ['row_id','physical_id','group','stage','split','geometry_sha256']} for r in job['rows']])
    key=signature(config)
    if (run/'config.json').exists() and signature(json.loads((run/'config.json').read_text()))!=key:raise ValueError('Run settings changed. Choose a new output directory.')
    write_json(run/'config.json',config)
    if a.preflight:
        r=next(r for r in rows if r['split']=='train');s,_=surface_data(r,job,a.output);model.eval()
        with T.no_grad():
            x=model(tensor_data(s,device),int(s['seed']));y=model(tensor_data(s,device),int(s['seed']))
        if not T.allclose(x,y,atol=a.probability_tolerance,rtol=0):raise RuntimeError('Backend is not repeatable in eval mode.')
        print(f'PASS {job["model"]}: logits {tuple(x.shape)}; genuine RGB={bool(s["has_rgb"])}',flush=True)
        write_json(run/'preflight.json',{'status':'passed','shape':list(x.shape),'max_repeat_difference':float((x-y).abs().max())});return
    if (run/'status.json').exists() and not a.final_evaluate:
        status=json.loads((run/'status.json').read_text())
        if status.get('status')=='complete' and a.resume and not a.evaluate_only:
            print('Already complete:',job['name'],flush=True);return
    sampling=[]
    for r in rows:
        s,_=surface_data(r,job,a.output)
        sampling.append({'row_id':r['row_id'],'scanner':r['scanner'],'sample_name':r['sample_name'],
                         'entries':len(s['coord']),'unique_points':len(np.unique(s['coord'],axis=0)),
                         'unique_source_faces':len(np.unique(s['face_index'][s['face_index']>=0])),
                         'effective_cells':int(s['effective_cells']),'display_rgb_available':bool(s['has_rgb']),
                         'training_rgb_nonzero':bool(s['color'].max()>0),'native_face_centres':job['model']=='tsgcnet'})
    write_csv(run/'sampling_qc.csv',sampling)
    start=time.perf_counter();features=embedding_cache(model,rows,job,a,device,provenance) if model.frozen else None
    extraction=time.perf_counter()-start
    if a.final_evaluate or a.evaluate_only:best=load_best(model,run/'best.pt',key,device);history=[]
    else:best,history=train(model,rows,job,a,device,run,key,features)
    parts=['test'] if a.final_evaluate else ['val']+(['assessment'] if job['protocol'] in ['cv','loso'] else [])
    for part in parts:
        selected=[r for r in rows if r['split']==part]
        if not selected:continue
        prob=predict(model,selected,job,a,device,features)
        m=export_evaluation(run,part,selected,prob,a.bootstrap)
        m.update(best_epoch=best,feature_preparation_seconds=extraction,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),device=str(device))
        write_json(run/f'metrics_{part}.json',m)
        if not a.final_evaluate and part==('assessment' if job['protocol']!='reference' else 'val'):
            robust=[]
            perturbations=['rotate_z_90','translate_scaled_0p1']
            if job['protocol']=='reference' and job['variant'] in ['geometry_rgb','rgb_combined'] and job['seed']==a.seeds[0]:
                perturbations+=['rotate_z_180','rotate_z_270','point_order','shuffle_rgb','gray']
            for perturb in perturbations:
                pp=predict(model,selected,job,a,device,features=None,perturb=perturb)
                robust.append({'transform':perturb,'changed_predictions':int(sum(pp.argmax(1)!=prob.argmax(1))),
                               'mean_probability_difference':float(np.abs(pp-prob).mean()),**score(selected,pp)})
            write_json(run/f'rigid_sensitivity_{part}.json',robust)
            write_json(run/f'input_sensitivity_{part}.json',robust)
            # Explain only reference baseline first seed; other runs retain predictions.
            if a.focus_examples and job['protocol']=='reference' and job['variant'] in ['area_geometry','geometry_rgb','rgb_combined'] and job['seed']==a.seeds[0]:
                focus_outputs(model,selected,prob,job,a,device,run,features)
    if history:learning_plot(history,run/'training.png')
    previous=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
    write_json(run/'status.json',{'status':'evaluated_checkpoint' if a.evaluate_only else 'complete','best_epoch':best,
                                 'completed_epochs':history[-1]['epoch'] if history else previous.get('completed_epochs'),
                                 'final_test_evaluated':bool(a.final_evaluate or previous.get('final_test_evaluated'))})
    del model;gc.collect()
    if device.type=='cuda':T.cuda.empty_cache()

def main(argv=None):
    global T
    a=arguments(argv);a.output=a.output.expanduser().resolve();a.output.mkdir(parents=True,exist_ok=True)
    if a.report_only:rebuild_report(a.output);return 0
    rows,audit,excluded=load_snapshot(a.data_root,a.allow_incomplete)
    reference=reference_split(rows,a.split_seed,a.split_json)
    jobs=plan_jobs(reference,a)
    frozen_test={r['physical_id'] for r in reference if r['split']=='test'}
    for job in jobs:
        if job['protocol']!='reference' and frozen_test&{r['physical_id'] for r in job['rows']}:raise AssertionError('Reserved test subjects entered an investigation.')
    identity_signature=signature([audit['index_sha256'],[{k:r[k] for k in ['row_id','geometry_sha256','split']} for r in reference]])
    stamp=a.output/'dataset_identity.json'
    if stamp.exists() and json.loads(stamp.read_text())['signature']!=identity_signature:raise ValueError('This output directory belongs to a different snapshot or split. Use a new output.')
    write_json(stamp,{'signature':identity_signature,'root':str(a.data_root),'version':VERSION})
    write_json(a.output/'cohort_audit.json',audit);write_csv(a.output/'excluded_scans.csv',excluded);write_csv(a.output/'scan_manifest.csv',reference)
    write_json(a.output/'reference_split.json',{'physical_ids':{p:sorted({r['physical_id'] for r in reference if r['split']==p}) for p in ['train','val','test']}})
    write_csv(a.output/'experiment_plan.csv',[{k:v for k,v in j.items() if k!='rows'}|{'train_scans':sum(r['split']=='train' for r in j['rows']),
                  'val_scans':sum(r['split']=='val' for r in j['rows']),'assessment_scans':sum(r['split']=='assessment' for r in j['rows'])} for j in jobs])
    for p in ['train','val','test']:
        rr=[r for r in reference if r['split']==p]
        print(f'{p}: {len(rr)} scans; {len({r["physical_id"] for r in rr})} specimens; {len({r["physical_id"] for r in rr if r["stage"]=="partial"})} partial specimens',flush=True)
    print(f'Planned {len(jobs)} model jobs; report: {a.output/"comparison.html"}',flush=True)
    if a.plan:rebuild_report(a.output);return 0
    # Test surfaces are not opened for QC or controls until explicit final evaluation.
    dev=[r for r in reference if r['split']!='test']
    if not a.final_evaluate:
        metadata=metadata_table(dev,a.output)
        inspect_gallery(dev,a.inspect_samples,a.output)
        if a.audit_only:rebuild_report(a.output);return 0
        protocols={}
        for j in jobs:protocols.setdefault(j['protocol']+'_'+j['tag'],[r for r in j['rows'] if r['split']!='test'])
        for name,rr in protocols.items():baseline_controls(rr,metadata,a.output/'controls'/name,a.bootstrap)
        if a.controls_only:rebuild_report(a.output);return 0
    import torch
    T=torch
    if a.device=='auto':device=T.device('cuda' if T.cuda.is_available() else 'cpu')
    else:device=T.device(a.device)
    if device.type=='cpu' and not a.allow_cpu:raise RuntimeError('CUDA unavailable. Activate your working biocal3d environment or explicitly pass --allow-cpu.')
    a.backends_root=resolve_backends(a.backends_root)
    print(f'Device: {device}; torch {T.__version__}; official backends: {a.backends_root}',flush=True)
    if a.preflight:
        selected=[];seen=set()
        for j in jobs:
            if j['variant']=='area_geometry' and j['protocol']=='reference' and j['model'] not in seen:selected.append(j);seen.add(j['model'])
        jobs=selected
    failed=[]
    for i,job in enumerate(jobs,1):
        print(f'\n[{i}/{len(jobs)}] {job["name"]}',flush=True)
        run=a.output/'runs'/job['name'];run.mkdir(parents=True,exist_ok=True)
        try:run_job(job,a,device)
        except KeyboardInterrupt:
            write_json(run/'status.json',{'status':'interrupted','note':'Last completed epoch is in last.pt. Resume with unchanged settings.'});rebuild_report(a.output)
            print('Interrupted; saved completed epochs and refreshed report. Use --resume.',flush=True);return 130
        except Exception as e:
            failed.append({'job':job['name'],'error':str(e)});write_json(run/'status.json',{'status':'failed','error':str(e)})
            (run/'error.txt').write_text(traceback.format_exc(),encoding='utf-8');print('FAILED:',str(e),flush=True)
        finally:
            gc.collect()
            if device.type=='cuda':T.cuda.empty_cache()
            rebuild_report(a.output)
    write_json(a.output/'failures.json',failed)
    print(f'Finished: {len(jobs)-len(failed)} successful / {len(jobs)} jobs. Report: {a.output/"comparison.html"}',flush=True)
    return 1 if failed else 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as e:print(f'ERROR: {e}',file=sys.stderr,flush=True);raise SystemExit(1)
