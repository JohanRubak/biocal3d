"""BioCal3D frozen-DINO comparison and stage-pretraining ablation.

Dependencies: torch, torchvision (compatible with your GPU), numpy, pandas,
matplotlib, scikit-learn. Uses existing scan_manifest.csv and maps.npz.
Run --help for options; no mesh preparation is repeated.

Start with classification (PowerShell, one line):
python biocal3d_dino_ablation.py --mode classification --dinov3-repo C:/path/to/dinov3 --dinov3-weights C:/path/to/dinov3_vits16_pretrain.pth --seeds 42 --output C:/path/to/classification_study

Full six-way study (use a different output folder):
python biocal3d_dino_ablation.py --mode full --labels-root C:/path/to/verified_labels --dinov3-repo C:/path/to/dinov3 --dinov3-weights C:/path/to/dinov3_vits16_pretrain.pth --output C:/path/to/full_study

Preflight your installed dependencies and trainable models:
python biocal3d_dino_ablation.py --self-test

DINOv3 weights must be the official pretrained ViT-S/16 checkpoint, not S+.
Clone the official DINOv3 repo and install its runtime requirements in your
biocal3d environment. Pin repo commits for reproducibility. If CUDA operations
are unsupported on your GPU, use --device cpu (slower). Feature extraction is
one scan at a time; try --batch-size 1 --width 32 if decoder memory is tight.
This does not reproduce the paper's complete training protocol: decoder width,
ROI masking, sparse-label supervision and classification pretraining are
BioCal3D adaptations. Default 256 input reduces detail relative to old 518 maps;
use --size 518 with a new output directory for a higher-resolution experiment.
No HD95 is reported because patch annotations cannot establish boundary truth.
Pretraining uses additional stage-labelled scans and compute; gains therefore
measure the practical value of that extra supervision and initialization.
Empty-foreground Dice/IoU is 1 for empty prediction, otherwise 0. Undefined
precision/recall is NaN; inspect positive/labelled pixel counts in predictions.

Models: DINOv2-S/14-register + simple head; DINOv3-S/16 + simple head;
DINOv3-S/16 + TPA/SAD decoder adapted from the authors' released SegDINOv2.
Every encoder is frozen. Classification trains spatial layers that are retained
for segmentation. Direct runs start from the identical spatial initialization.
Segmentation outputs are freshly initialized identically in both routes.

Labels: --labels-root/{train,val}/{scanner}/{sample_name}/pseudo_labels.npz
with labels_patch, verified, patch_valid and identity fields from your editor.
Alternatively labels.npz with labels (full map resolution), verified=True,
and identity fields split/scanner/sample_name/physical_id. Values -1/0/1.
Unknown areas and invalid ROI pixels never enter loss or metrics. Patch labels
are expanded nearest-neighbour: results measure coarse annotation agreement,
not precise boundary accuracy. Never treat stage labels as pixel labels.

Qualitative exports: classification_qualitative/*.png and *.npz include every
stage Grad-CAM; classification_focus_comparison compares models per image.
segmentation_qualitative includes predictions, labels, dense-output Grad-CAM,
and paired classification focus before/after segmentation (pretrained route).
The retained classifier after segmentation is a diagnostic fixed-head probe;
it is no longer calibrated/trained for the adapted spatial features.
CAM explains sensitivity of the selected score, not a plaque annotation.
Before/after CAM uses a common scale per class; cross-model CAM is independently
normalized because raw gradient magnitudes are not comparable across networks.
Use your original command plus --qualitative-only to export saved checkpoints
without retraining; missing runs are skipped. Original experiment settings must
match. Updated plotting code can reuse prior configurations/checkpoints.

No test-set inference. Validation metrics are exploratory. No augmentation is
used because encoder features are cached; all runs use the same input pixels.
DINOv2 pads the common input to a multiple of 14; DINOv3 to a multiple of 16.
Sources:
https://papers.miccai.org/miccai-2026/paper/0067_paper.pdf
https://github.com/script-Yang/segdino_v2/blob/main/dpt.py (Apache-2.0)
https://github.com/facebookresearch/dinov3
"""
from __future__ import annotations
import argparse, copy, hashlib, json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

STAGES = ['baseline', 'partial', 'clean']
MODELS = ['dinov2_simple', 'dinov3_simple', 'dinov3_segdino']
ROOT = Path(r'C:\Users\johan\repos\biocal3d\data\02-09-2026-Data collected')


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resize(x, size, mode='bilinear'):
    return F.interpolate(x, size=size, mode=mode, **({'align_corners': False} if mode == 'bilinear' else {}))


def map_data(row, size):
    with np.load(Path(row['map_dir']) / 'maps.npz', allow_pickle=False) as z:
        rgb = z['rgb'].astype(np.float32); valid = z['valid'].astype(bool)
    if rgb.shape[:2] != valid.shape or rgb.shape[-1] != 3 or not np.isfinite(rgb).all():
        raise ValueError('Invalid map: ' + row['map_dir'])
    if rgb.min() < 0 or rgb.max() > 1.001: raise ValueError('Expected RGB in [0,1]')
    rgb = resize(torch.from_numpy(rgb.transpose(2,0,1))[None], (size,size))[0]
    valid = resize(torch.from_numpy(valid.astype(np.float32))[None,None], (size,size), 'nearest')[0,0].bool()
    if not valid.any(): raise ValueError('Empty valid ROI')
    return rgb, valid


def load_label(row, root, size, valid):
    if root is None: return torch.full((size,size), -1, dtype=torch.long)
    folder = root / row['split'] / row['scanner'] / row['sample_name']
    paths = [p for p in [folder/'labels.npz', folder/'pseudo_labels.npz'] if p.exists()]
    if len(paths)>1: raise ValueError(f'Ambiguous annotations: {folder}')
    if not paths: return torch.full((size,size), -1, dtype=torch.long)
    with np.load(paths[0], allow_pickle=False) as z:
        if 'verified' not in z or not bool(z['verified'].item()):
            return torch.full((size,size), -1, dtype=torch.long)
        for field in ['split','scanner','sample_name','physical_id']:
            if field not in z or str(z[field].item()) != str(row[field]):
                raise ValueError(f'Annotation identity mismatch {field}: {paths[0]}')
        key = 'labels' if 'labels' in z else 'labels_patch'
        label = z[key].copy()
        if label.ndim != 2 or not np.isin(label, [-1,0,1]).all(): raise ValueError(f'Invalid labels: {paths[0]}')
        if key == 'labels_patch':
            if 'patch_valid' not in z or z['patch_valid'].shape != label.shape: raise ValueError('Missing patch_valid')
            label[~z['patch_valid'].astype(bool)] = -1
        else:
            with np.load(Path(row['map_dir'])/'maps.npz') as m:
                if label.shape != m['valid'].shape: raise ValueError('Full labels must match original maps')
    label = resize(torch.from_numpy(label.astype(np.float32))[None,None], (size,size), 'nearest')[0,0].long()
    label[~valid] = -1
    return label


class Refine(nn.Module):
    def __init__(self, c):
        super().__init__()
        groups = next(g for g in range(min(c,32),0,-1) if c%g==0)
        self.net = nn.Sequential(nn.Conv2d(c,c,3,padding=1,groups=c,bias=False),
            nn.Conv2d(c,c,1,bias=False), nn.GroupNorm(groups,c), nn.GELU())
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self,x): return x + self.gamma*self.net(x)


class SpatialModel(nn.Module):
    def __init__(self, channels, width, pyramid=False, tpa=True, sad=True, patch=1):
        super().__init__(); self.pyramid=pyramid; self.tpa=tpa; self.sad=sad; self.patch=patch
        if pyramid:
            self.project = nn.ModuleList([nn.Conv2d(channels,width,1,bias=False) for _ in range(4)])
            self.resample = nn.ModuleList([nn.Conv2d(width,width,3,padding=1,bias=False) for _ in range(4)])
            self.intra = nn.ModuleList([Refine(width) for _ in range(4)])
            self.inter = nn.ModuleList([Refine(width) for _ in range(4)])
        else: self.simple = nn.Sequential(nn.Conv2d(channels,width,1),nn.ReLU())
        self.classifier = nn.Linear(width,3)
        self.segmenter = nn.Conv2d(width,2,1)
    def spatial(self, features):
        if not self.pyramid: return self.simple(features[-1])
        levels=[]
        for i,x in enumerate(features):
            x=self.project[i](x)
            # Released SegDINOv2 uses patch-grid factors 8,4,2,1.
            if self.tpa:
                factor=[8,4,2,1][i]
                x=self.resample[i](resize(x,(x.shape[-2]*factor,x.shape[-1]*factor)))
            if self.sad: x=self.intra[i](x)
            levels.append(x)
        x=levels[-1]
        if self.sad: x=self.inter[-1](x)
        for i in [2,1,0]:
            x=resize(x,levels[i].shape[-2:])+levels[i]
            if self.sad: x=self.inter[i](x)
        return x
    def forward(self,features,valid,task):
        x=self.spatial(features)
        if task=='classification':
            # Area pooling retains valid ROI fractions on a coarse feature grid.
            pad_h=(-valid.shape[-2])%self.patch; pad_w=(-valid.shape[-1])%self.patch
            mask=F.adaptive_avg_pool2d(F.pad(valid[:,None].float(),(0,pad_w,0,pad_h)),x.shape[-2:])
            pooled=(x*mask).sum((2,3))/mask.sum((2,3)).clamp_min(1e-8)
            return self.classifier(pooled)
        h,w=valid.shape[-2:]; ph=(-h)%self.patch; pw=(-w)%self.patch
        return resize(self.segmenter(x),(h+ph,w+pw))[...,:h,:w]


class CachedDataset(Dataset):
    def __init__(self, records, cache, labels, size, segmentation=False):
        self.records=[]; self.cache=cache; self.labels=labels; self.size=size
        for row in records:
            _,valid=map_data(row,size); target=load_label(row,labels,size,valid)
            if not segmentation or (target>=0).any(): self.records.append(row)
    def __len__(self): return len(self.records)
    def __getitem__(self,i):
        row=self.records[i]; _,valid=map_data(row,self.size)
        z=torch.load(self.cache/f"{row['row_id']}.pt",map_location='cpu',weights_only=True)
        return [f.float() for f in z['features']],valid,STAGES.index(row['stage']),load_label(row,self.labels,self.size,valid),i


def cache_features(records, version, args, device):
    repo=args.dinov2_repo if version=='dinov2' else args.dinov3_repo
    weights=None if version=='dinov2' else args.dinov3_weights
    if version=='dinov3' and not weights: raise ValueError('Supply --dinov3-weights (local pretrained S/16 checkpoint).')
    name='dinov2_vits14_reg' if version=='dinov2' else 'dinov3_vits16'
    patch=14 if version=='dinov2' else 16
    repo_path=Path(repo)
    repo_identity=repo
    if repo_path.is_dir():
        import subprocess
        result=subprocess.run(['git','-C',str(repo_path),'rev-parse','HEAD'],capture_output=True,text=True)
        repo_identity += ':'+result.stdout.strip()
    spec={'model':name,'repo':repo_identity,'weights':digest(weights) if weights else 'hub_pretrained',
          'size':args.size,'layers':[2,5,8,11],'normalization':'imagenet','format':1}
    cache=args.output/'feature_cache'/version
    cache.mkdir(parents=True,exist_ok=True)
    config=cache/'config.json'
    if config.exists() and json.loads(config.read_text()) != spec: raise ValueError(f'Cache settings changed: {cache}; use a new output.')
    config.write_text(json.dumps(spec,indent=2))
    pending=[]
    for row in records:
        sha=digest(Path(row['map_dir'])/'maps.npz'); dest=cache/f"{row['row_id']}.pt"
        if dest.exists():
            saved=torch.load(dest,map_location='cpu',weights_only=True)
            if saved['map_sha256']==sha: continue
        pending.append((row,sha,dest))
    if pending:
        kw={'source':'local' if repo_path.is_dir() else 'github'}
        if version=='dinov3': kw['weights']=str(weights)
        else: kw['pretrained']=True
        encoder=torch.hub.load(repo,name,**kw).to(device).eval()
        for p in encoder.parameters(): p.requires_grad=False
        mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
        std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
        for i,(row,sha,dest) in enumerate(pending,1):
            rgb,_=map_data(row,args.size); x=(rgb[None].to(device)-mean)/std
            pad=(-args.size)%patch; x=F.pad(x,(0,pad,0,pad))
            with torch.no_grad(): features=encoder.get_intermediate_layers(x,n=[2,5,8,11],reshape=True)
            if len(features)!=4 or any(f.ndim!=4 for f in features): raise RuntimeError('Unexpected intermediate feature format')
            torch.save({'features':[f[0].cpu().half() for f in features],'map_sha256':sha},dest)
            print(f'{version} features {i}/{len(pending)}: {row["scanner"]} {row["sample_name"]}',flush=True)
        del encoder
        if device.type=='cuda': torch.cuda.empty_cache()
    return cache


def pixel_metrics(prob,label):
    keep=label>=0; truth=label[keep]==1; pred=prob[keep]>=.5
    tp=int((truth&pred).sum()); fp=int((~truth&pred).sum()); fn=int((truth&~pred).sum())
    def ratio(a,b,empty=1.): return a/b if b else empty
    return dict(dice=ratio(2*tp,2*tp+fp+fn),iou=ratio(tp,tp+fp+fn),
        precision=ratio(tp,tp+fp,float('nan')),recall=ratio(tp,tp+fn,float('nan')),
        labelled_pixels=int(keep.sum()),gt_positive_pixels=int(truth.sum()))


def evaluate(model,loader,device,task):
    model.eval(); loss_sum=0.; count=0; truths=[]; preds=[]; metrics=[]
    with torch.no_grad():
        for fs,valid,stage,label,idx in loader:
            fs=[f.to(device) for f in fs]; valid=valid.to(device); label=label.to(device)
            output=model(fs,valid,task)
            if task=='classification':
                loss=F.cross_entropy(output,stage.to(device),reduction='none')
                truths.extend(stage.tolist()); preds.extend(output.argmax(1).cpu().tolist())
                loss_sum+=loss.sum().item(); count+=len(stage)
            else:
                for j in range(len(idx)):
                    keep=label[j]>=0
                    loss_sum+=F.cross_entropy(output[j:j+1],label[j:j+1],ignore_index=-1).item(); count+=1
                    metrics.append(pixel_metrics(output[j].softmax(0)[1].cpu().numpy(),label[j].cpu().numpy()))
    if not count: raise ValueError(f'No evaluation examples for {task}')
    score=balanced_accuracy_score(truths,preds) if task=='classification' else float(np.mean([m['dice'] for m in metrics]))
    return loss_sum/count,score,truths,preds


def train(model,trainset,valset,device,task,epochs,args,out,seed):
    if not len(trainset) or not len(valset): raise ValueError(f'{task} needs nonempty training and validation sets')
    seed_all(seed)
    generator=torch.Generator().manual_seed(seed)
    tr=DataLoader(trainset,batch_size=args.batch_size,shuffle=True,generator=generator)
    va=DataLoader(valset,batch_size=args.batch_size)
    optim=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    best=float('inf'); state=None; history=[]; start=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train(); total=0.; n=0
        for fs,valid,stage,label,idx in tr:
            fs=[f.to(device) for f in fs]; valid=valid.to(device); label=label.to(device)
            optim.zero_grad(set_to_none=True); output=model(fs,valid,task)
            if task=='classification': loss=F.cross_entropy(output,stage.to(device))
            else:
                # Equal scan weighting prevents dense annotations dominating sparse ones.
                loss=torch.stack([F.cross_entropy(output[j:j+1],label[j:j+1],ignore_index=-1) for j in range(len(idx))]).mean()
            loss.backward(); optim.step(); total+=loss.item()*len(idx); n+=len(idx)
        vl,score,_,_=evaluate(model,va,device,task)
        history.append(dict(epoch=epoch,train_loss=total/n,val_loss=vl,val_score=score))
        if vl<best:
            best=vl; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save({'state_dict':state,'task':task,'epoch':epoch,'val_loss':vl},out/f'{task}_best.pt')
        print(f'{out.name} {task} {epoch}/{epochs}: train={total/n:.4f} val={vl:.4f} score={score:.4f}',flush=True)
        pd.DataFrame(history).to_csv(out/f'{task}_history.csv',index=False)
    model.load_state_dict(state)
    fig,axes=plt.subplots(1,2,figsize=(9,3)); h=pd.DataFrame(history)
    axes[0].plot(h.epoch,h.train_loss,label='train'); axes[0].plot(h.epoch,h.val_loss,label='val'); axes[0].legend()
    axes[1].plot(h.epoch,h.val_score); axes[1].set_title('Balanced accuracy' if task=='classification' else 'Mean scan Dice')
    fig.tight_layout(); fig.savefig(out/f'{task}_history.png',dpi=150); plt.close(fig)
    return time.perf_counter()-start


def export_predictions(model,dataset,device,out,task):
    model.eval(); rows=[]; dest=out/f'{task}_predictions'; dest.mkdir(exist_ok=True)
    with torch.no_grad():
        for i,row in enumerate(dataset.records):
            fs,valid,stage,label,_=dataset[i]
            output=model([f[None].to(device) for f in fs],valid[None].to(device),task)[0]
            rec={k:row[k] for k in ['row_id','scanner','sample_name','physical_id','stage']}
            if task=='classification':
                probs=output.softmax(0).cpu().numpy(); rec['prediction']=STAGES[int(probs.argmax())]
                rec.update({f'p_{s}':float(p) for s,p in zip(STAGES,probs)})
            else:
                prob=output.softmax(0)[1].cpu().numpy(); lab=label.numpy(); mask=valid.numpy()
                rec.update(pixel_metrics(prob,lab)); rec['coverage'] = float((prob[mask]>=.5).mean())
                np.savez_compressed(dest/f"{row['row_id']}.npz",probability=prob,labels=lab,valid=mask)
            rows.append(rec)
    df=pd.DataFrame(rows); df.to_csv(out/f'{task}_predictions.csv',index=False)
    if task=='classification':
        cm=confusion_matrix(df.stage,df.prediction,labels=STAGES)
        fig,ax=plt.subplots(); ax.imshow(cm,cmap='Blues'); ax.set_xticks(range(3),STAGES); ax.set_yticks(range(3),STAGES)
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        for i in range(3):
            for j in range(3): ax.text(j,i,str(cm[i,j]),ha='center',va='center')
        fig.tight_layout(); fig.savefig(out/'classification_confusion.png',dpi=150); plt.close(fig)
    return df


def focus_maps(model, features, valid):
    """Grad-CAM at shared spatial output; encoder remains frozen.

    Stage CAM uses each class logit. Segmentation CAM uses the valid-ROI mean
    plaque-minus-background logit. The latter explains a dense output score,
    and is different from the segmentation probability map.
    """
    model.eval()
    with torch.no_grad(): activation=model.spatial(features)
    activation=activation.detach().requires_grad_(True)
    h,w=valid.shape[-2:]; ph=(-h)%model.patch; pw=(-w)%model.patch
    mask=F.adaptive_avg_pool2d(F.pad(valid[:,None].float(),(0,pw,0,ph)),activation.shape[-2:])
    logits=model.classifier((activation*mask).sum((2,3))/mask.sum((2,3)).clamp_min(1e-8))
    dense=resize(model.segmenter(activation),(h+ph,w+pw))[...,:h,:w]
    scores=[logits[:,i].sum() for i in range(3)]
    scores.append(((dense[:,1]-dense[:,0])*valid).sum()/valid.sum().clamp_min(1))
    maps=[]
    for score in scores:
        gradient=torch.autograd.grad(score,activation,retain_graph=True)[0]
        weights=(gradient*mask).sum((2,3),keepdim=True)/mask.sum((2,3),keepdim=True).clamp_min(1e-8)
        cam=torch.relu((weights*activation).sum(1,keepdim=True))
        cam=resize(cam,(h+ph,w+pw))[...,:h,:w]*valid[:,None]
        maps.append(cam[0,0].detach().cpu().numpy())
    return np.stack(maps),logits.softmax(1)[0].detach().cpu().numpy(),dense.softmax(1)[0,1].detach().cpu().numpy()


def draw_heat(ax,rgb,heat,valid,title,vmax=None):
    ax.imshow(rgb)
    vmax=float(np.max(heat)) if vmax is None else float(vmax)
    ax.imshow(np.ma.masked_where(~valid,heat),cmap='inferno',vmin=0,vmax=max(vmax,1e-12),alpha=.60)
    ax.set_title(title + ('\n(no positive CAM)' if vmax<=1e-12 else ''),fontsize=9); ax.axis('off')


def export_qualitative(model,dataset,device,out,task,before=None):
    destination=out/f'{task}_qualitative'; destination.mkdir(exist_ok=True)
    records=[]
    for i,row in enumerate(dataset.records):
        fs,valid,stage,label,_=dataset[i]
        raw,probs,plaque=focus_maps(model,[f[None].to(device) for f in fs],valid[None].to(device))
        rgb,_=map_data(row,dataset.size); image=rgb.permute(1,2,0).numpy(); mask=valid.numpy()
        payload=dict(stage_cam_raw=raw[:3],segmentation_cam_raw=raw[3],stage_probabilities=probs,
                     plaque_probability=plaque,valid=mask,labels=label.numpy())
        np.savez_compressed(destination/f"{row['row_id']}.npz",**payload)
        before_path=before/f"{row['row_id']}.npz" if before is not None else None
        prior=None
        if before_path is not None and before_path.exists():
            with np.load(before_path) as z: prior={k:z[k].copy() for k in z.files}
        if task=='classification':
            fig,axes=plt.subplots(1,4,figsize=(15,4)); axes[0].imshow(image); axes[0].set_title('RGB'); axes[0].axis('off')
            for j,stage_name in enumerate(STAGES):
                draw_heat(axes[j+1],image,raw[j],mask,f'{stage_name} Grad-CAM | p={probs[j]:.3f}')
            fig.suptitle(f"{row['scanner']} {row['sample_name']} | actual {row['stage']} | predicted {STAGES[int(probs.argmax())]}")
        else:
            fig,axes=plt.subplots(2,4,figsize=(15,8)); axes=axes.ravel()
            axes[0].imshow(image); axes[0].set_title('RGB'); axes[0].axis('off')
            axes[1].imshow(image); lab=label.numpy()
            axes[1].imshow(np.ma.masked_where(lab<0,lab),vmin=0,vmax=1,cmap='coolwarm',alpha=.55)
            axes[1].set_title('Verified labels; unknown transparent'); axes[1].axis('off')
            draw_heat(axes[2],image,plaque,mask,'Plaque probability after segmentation',vmax=1)
            draw_heat(axes[3],image,raw[3],mask,'Segmentation Grad-CAM (ROI mean logit)')
            # Same stage query and fixed classifier before/after; common CAM scale.
            for j,stage_name in enumerate(['baseline','partial']):
                k=STAGES.index(stage_name)
                scale=max(float(raw[k].max()),float(prior['stage_cam_raw'][k].max())) if prior else float(raw[k].max())
                if prior:
                    draw_heat(axes[4+2*j],image,prior['stage_cam_raw'][k],mask,f'{stage_name} focus BEFORE',scale)
                    difference=raw[k]-prior['stage_cam_raw'][k]
                    records.append(dict(row_id=row['row_id'],scanner=row['scanner'],sample_name=row['sample_name'],stage_query=stage_name,
                        mean_abs_raw_cam_change=float(np.abs(difference[mask]).mean())))
                else:
                    axes[4+2*j].text(.5,.5,'No classification pretraining',ha='center',va='center'); axes[4+2*j].axis('off')
                draw_heat(axes[5+2*j],image,raw[k],mask,f'{stage_name} fixed-head probe AFTER',scale)
            fig.suptitle(f"{row['scanner']} {row['sample_name']} | stage probe after segmentation is diagnostic, not a trained stage predictor")
        fig.tight_layout(); fig.savefig(destination/f"{row['row_id']}.png",dpi=140); plt.close(fig)
        if task=='segmentation' and prior is not None:
            fig,axes=plt.subplots(2,4,figsize=(15,7))
            for line in range(2):
                axes[line,0].imshow(image); axes[line,0].set_title('Before' if line==0 else 'After; fixed-head probe'); axes[line,0].axis('off')
                for j,stage_name in enumerate(STAGES):
                    scale=max(float(prior['stage_cam_raw'][j].max()),float(raw[j].max()))
                    heat=prior['stage_cam_raw'][j] if line==0 else raw[j]
                    draw_heat(axes[line,j+1],image,heat,mask,stage_name+' Grad-CAM',scale)
            fig.suptitle(f"{row['scanner']} {row['sample_name']} | same stage head; shared before/after scale per class")
            fig.tight_layout(); fig.savefig(destination/f"{row['row_id']}_all_stage_focus.png",dpi=140); plt.close(fig)
    if records: pd.DataFrame(records).to_csv(destination/'focus_changes.csv',index=False)
    print(f'Exported {task} qualitative views: {destination}',flush=True)


def qualitative_comparison(output,records):
    """Same image and stage queries across architectures and routes, per seed."""
    runs=sorted((output/'runs').glob('*'))
    seeds=sorted({p.name.rsplit('__',1)[-1] for p in runs if '__' in p.name})
    destination=output/'classification_focus_comparison'; destination.mkdir(exist_ok=True)
    for seed in seeds:
        selected=[p for p in runs if p.name.endswith('__'+seed) and (p/'classification_qualitative').exists()]
        if not selected: continue
        for row in records:
            if row['split']!='val': continue
            entries=[]
            for path in selected:
                file=path/'classification_qualitative'/f"{row['row_id']}.npz"
                if file.exists():
                    with np.load(file) as z: entries.append((path.name,z['stage_cam_raw'].copy(),z['valid'].copy()))
            if not entries: continue
            # Read input at exported size, not at a hardcoded resolution.
            size=entries[0][2].shape[0]; rgb,_=map_data(row,size); image=rgb.permute(1,2,0).numpy()
            fig,axes=plt.subplots(len(entries),4,figsize=(14,3.5*len(entries)),squeeze=False)
            for i,(name,cams,mask) in enumerate(entries):
                axes[i,0].imshow(image); axes[i,0].set_title(name,fontsize=9); axes[i,0].axis('off')
                for j in range(3): draw_heat(axes[i,j+1],image,cams[j],mask,STAGES[j]+' Grad-CAM')
            fig.suptitle(f"{row['scanner']} {row['sample_name']} | each model CAM independently normalized; brightness is not comparable across models")
            fig.tight_layout(); fig.savefig(destination/f"{row['row_id']}__{seed}.png",dpi=130); plt.close(fig)


def segmentation_focus_comparison(output,records):
    runs=sorted((output/'runs').glob('*'))
    seeds=sorted({p.name.rsplit('__',1)[-1] for p in runs if '__' in p.name})
    destination=output/'segmentation_focus_comparison'; destination.mkdir(exist_ok=True)
    for seed in seeds:
        selected=[p for p in runs if p.name.endswith('__'+seed) and (p/'segmentation_qualitative').exists()]
        for row in records:
            if row['split']!='val': continue
            entries=[]
            for path in selected:
                file=path/'segmentation_qualitative'/f"{row['row_id']}.npz"
                if file.exists():
                    with np.load(file) as z: entries.append((path.name,{k:z[k].copy() for k in z.files}))
            if not entries: continue
            size=entries[0][1]['valid'].shape[0]; rgb,_=map_data(row,size); image=rgb.permute(1,2,0).numpy()
            fig,axes=plt.subplots(len(entries),4,figsize=(14,3.5*len(entries)),squeeze=False)
            for i,(name,z) in enumerate(entries):
                axes[i,0].imshow(image); axes[i,0].set_title(name,fontsize=9); axes[i,0].axis('off')
                draw_heat(axes[i,1],image,z['plaque_probability'],z['valid'],'Plaque probability',1)
                draw_heat(axes[i,2],image,z['segmentation_cam_raw'],z['valid'],'Segmentation Grad-CAM')
                query=STAGES.index(row['stage'])
                draw_heat(axes[i,3],image,z['stage_cam_raw'][query],z['valid'],row['stage']+' stage probe' if '__pretrained__' in name else 'Stage probe unavailable')
                if '__direct__' in name:
                    axes[i,3].clear(); axes[i,3].text(.5,.5,'No trained stage head',ha='center',va='center'); axes[i,3].axis('off')
            fig.suptitle(f"{row['scanner']} {row['sample_name']} | CAM independently normalized across runs")
            fig.tight_layout(); fig.savefig(destination/f"{row['row_id']}__{seed}.png",dpi=130); plt.close(fig)


def reports(output, records, size):
    tables=[]
    for path in sorted(output.glob('runs/*/segmentation_predictions.csv')):
        df=pd.read_csv(path); parts=path.parent.name.rsplit('__',2)
        df['model'],df['route'],df['seed']=parts[0],parts[1],int(parts[2]); tables.append(df)
    if not tables: return
    df=pd.concat(tables,ignore_index=True); df.to_csv(output/'all_segmentation_predictions.csv',index=False)
    # First average repeated scans within each independent specimen, then seeds.
    specimen=df.groupby(['model','route','seed','physical_id'])[['dice','iou','precision','recall']].mean().reset_index()
    specimen.to_csv(output/'specimen_metrics.csv',index=False)
    seeds=specimen.groupby(['model','route','seed'])[['dice','iou','precision','recall']].mean().reset_index()
    seeds.to_csv(output/'seed_metrics.csv',index=False)
    ranked=seeds.groupby(['model','route']).agg(dice_mean=('dice','mean'),dice_seed_sd=('dice','std'),iou_mean=('iou','mean'),n_seeds=('seed','nunique')).reset_index().sort_values('dice_mean',ascending=False)
    ranked.to_csv(output/'ranked_comparison.csv',index=False)
    df.groupby(['model','route','seed','scanner'])[['dice','iou']].mean().to_csv(output/'scanner_metrics.csv')
    df.groupby(['model','route','seed','stage'])[['dice','iou']].mean().to_csv(output/'stage_metrics.csv')
    paired=seeds.pivot(index=['model','seed'],columns='route',values='dice')
    if {'direct','pretrained'}<=set(paired.columns):
        paired['pretraining_gain']=paired.pretrained-paired.direct
        paired.to_csv(output/'pretraining_ablation.csv')
    fig,ax=plt.subplots(figsize=(10,4)); ax.bar(range(len(ranked)),ranked.dice_mean)
    ax.set_xticks(range(len(ranked)),[f'{r.model}\n{r.route}' for r in ranked.itertuples()],rotation=35,ha='right')
    ax.set_ylabel('Specimen-macro validation Dice'); ax.set_ylim(0,1); fig.tight_layout()
    fig.savefig(output/'comparison.png',dpi=180); plt.close(fig)
    first_seed=int(df.seed.min()); combinations=df[df.seed==first_seed][['model','route']].drop_duplicates().values.tolist()
    gallery=output/'matched_predictions'; gallery.mkdir(exist_ok=True)
    for row in records:
        if row['split']!='val': continue
        paths=[output/'runs'/f'{m}__{r}__{first_seed}'/'segmentation_predictions'/f"{row['row_id']}.npz" for m,r in combinations]
        if not all(p.exists() for p in paths): continue
        rgb,_=map_data(row,size); image=rgb.permute(1,2,0).numpy()
        fig,axes=plt.subplots(2,4,figsize=(15,7)); axes=axes.ravel()
        axes[0].imshow(image); axes[0].set_title('RGB')
        with np.load(paths[0]) as z: label=z['labels'].copy()
        axes[1].imshow(image); axes[1].imshow(np.ma.masked_where(label<0,label),vmin=0,vmax=1,cmap='coolwarm',alpha=.55); axes[1].set_title('Verified labels; unknown transparent')
        for ax,path,(m,r) in zip(axes[2:],paths,combinations):
            with np.load(path) as z: prob=z['probability']; valid=z['valid']
            ax.imshow(image); ax.imshow(np.ma.masked_where(~valid,prob),vmin=0,vmax=1,cmap='inferno',alpha=.55); ax.set_title(f'{m}\n{r}')
        for ax in axes: ax.axis('off')
        fig.suptitle(f"{row['scanner']} {row['sample_name']} | seed {first_seed}"); fig.tight_layout()
        fig.savefig(gallery/f"{row['row_id']}.png",dpi=130); plt.close(fig)


def self_test():
    seed_all(1); features=[torch.randn(2,12,4,4) for _ in range(4)]
    valid=torch.ones(2,32,32,dtype=torch.bool); valid[:,:3]=False
    for pyramid in [False,True]:
        model=SpatialModel(12,8,pyramid); initial=copy.deepcopy(model.state_dict())
        loss=F.cross_entropy(model(features,valid,'classification'),torch.tensor([0,2])); loss.backward()
        optim=torch.optim.SGD(model.parameters(),lr=.1); optim.step()
        shared=[k for k in initial if not k.startswith(('classifier.','segmenter.'))]
        assert any(not torch.equal(initial[k],model.state_dict()[k]) for k in shared)
        assert torch.equal(initial['segmenter.weight'],model.segmenter.weight)
        model.zero_grad(); logits=model(features,valid,'segmentation')
        label=torch.zeros(2,32,32,dtype=torch.long); label[:,:3]=-1
        F.cross_entropy(logits,label,ignore_index=-1).backward()
        assert logits.shape==(2,2,32,32)
        maps,probs,plaque=focus_maps(model,features,valid)
        assert maps.shape==(4,32,32) and probs.shape==(3,) and plaque.shape==(32,32)
        assert np.isfinite(maps).all() and np.all(maps[:,:3]==0)
        assert np.isclose(probs.sum(),1)
    assert pixel_metrics(np.zeros((2,2)),np.zeros((2,2)))['dice']==1
    assert pixel_metrics(np.ones((2,2)),np.zeros((2,2)))['dice']==0
    print('Self-test passed: shapes, gradients, shared-layer transfer, ignored labels, metrics.')


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--prep-root',type=Path,default=ROOT/'_dino_preparation_80_10_10')
    p.add_argument('--labels-root',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'_dino_ablation')
    p.add_argument('--models',nargs='+',choices=MODELS,default=MODELS)
    p.add_argument('--routes',nargs='+',choices=['direct','pretrained'],default=['direct','pretrained'])
    p.add_argument('--mode',choices=['classification','full'],default='full')
    p.add_argument('--dinov2-repo',default='facebookresearch/dinov2')
    p.add_argument('--dinov3-repo',default='facebookresearch/dinov3')
    p.add_argument('--dinov3-weights',type=Path)
    p.add_argument('--size',type=int,default=256)
    p.add_argument('--width',type=int,default=64,help='64 is a smaller BioCal3D adaptation; author config uses 256.')
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--classification-epochs',type=int,default=30)
    p.add_argument('--segmentation-epochs',type=int,default=50)
    p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--device',default='auto')
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--qualitative-only',action='store_true',help='Load saved best checkpoints and export figures without training.')
    args=p.parse_args()
    if args.self_test: self_test(); return
    if min(args.size,args.width,args.batch_size,args.classification_epochs,args.segmentation_epochs)<=0: p.error('Sizes/epochs must be positive')
    if args.mode=='full' and args.labels_root is None: p.error('--labels-root required for full mode; use --mode classification first')
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    manifest=pd.read_csv(args.prep_root/'scan_manifest.csv')
    required={'physical_id','split','scanner','sample_name','stage','map_dir'}
    if not required<=set(manifest): raise ValueError(f'Missing columns: {required-set(manifest)}')
    if manifest[list(required)].isna().any().any(): raise ValueError('Missing manifest values')
    if manifest.groupby('physical_id').split.nunique().max()>1: raise ValueError('Specimen leakage across splits')
    if manifest.duplicated(['scanner','sample_name']).any(): raise ValueError('Duplicate scans')
    if not set(manifest.stage)<=set(STAGES) or not set(manifest.split)<={'train','val','test'}: raise ValueError('Unknown stage/split')
    # Exclude test before opening any map or annotation.
    manifest=manifest[manifest.split.isin(['train','val'])].copy()
    records=manifest.to_dict('records')
    for row in records:
        folder=Path(row['map_dir'])
        if not folder.is_absolute(): folder=args.prep_root/folder
        row['map_dir']=str(folder)
        row['row_id']=hashlib.sha256(f"{row['scanner']}/{row['sample_name']}".encode()).hexdigest()[:16]
    if not {'train','val'}<=set(manifest.split): raise ValueError('Need train and val splits')
    for split in ['train','val']:
        if set(manifest.loc[manifest.split==split,'stage']) != set(STAGES):
            raise ValueError(f'{split} must contain baseline, partial and clean for this three-class comparison')
    # Preflight labels before downloading encoders; no silent fallback to weak labels.
    inventory=[]
    for row in records:
        _,valid=map_data(row,args.size); label=load_label(row,args.labels_root,args.size,valid)
        inventory.append({**{k:row[k] for k in ['scanner','sample_name','physical_id','split','stage']},'labelled_pixels':int((label>=0).sum())})
    inv=pd.DataFrame(inventory)
    if args.mode=='full' and any(not (inv.loc[inv.split==s,'labelled_pixels']>0).any() for s in ['train','val']):
        raise ValueError('Need verified segmentation annotations in BOTH train and val; use --mode classification until ready.')
    args.output.mkdir(parents=True,exist_ok=True)
    spec={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    spec['script_sha256']=digest(__file__); spec['torch_version']=torch.__version__
    spec['manifest_sha256']=digest(args.prep_root/'scan_manifest.csv')
    annotation_files=[]
    if args.labels_root:
        for row in records:
            folder=args.labels_root/row['split']/row['scanner']/row['sample_name']
            for filename in ['labels.npz','pseudo_labels.npz']:
                path=folder/filename
                if path.exists(): annotation_files.append({'path':str(path),'sha256':digest(path)})
    spec['annotations']=annotation_files
    lock=args.output/'experiment_config.json'
    if lock.exists():
        previous=json.loads(lock.read_text())
        ignored={'script_sha256','qualitative_only'}
        old={k:v for k,v in previous.items() if k not in ignored}
        new={k:v for k,v in spec.items() if k not in ignored}
        if old!=new: raise ValueError('Experiment settings changed; use the original arguments or a new --output directory.')
    elif args.qualitative_only: raise ValueError('--qualitative-only requires an existing experiment_config.json')
    if not args.qualitative_only: lock.write_text(json.dumps(spec,indent=2))
    inv.to_csv(args.output/'annotation_inventory.csv',index=False)
    for version in dict.fromkeys(m.split('_')[0] for m in args.models):
        if args.qualitative_only and not any((args.output/'runs'/f'{name}__pretrained__{seed}'/'classification_best.pt').exists() or (args.output/'runs'/f'{name}__direct__{seed}'/'segmentation_best.pt').exists() for name in args.models if name.startswith(version) for seed in args.seeds):
            print(f'No saved checkpoints for {version}; skipping qualitative export.',flush=True)
            continue
        cache=cache_features(records,version,args,device)
        datasets={}
        for task in ['classification','segmentation']:
            if task=='segmentation' and args.mode!='full': continue
            datasets[task]={s:CachedDataset([r for r in records if r['split']==s],cache,args.labels_root,args.size,task=='segmentation') for s in ['train','val']}
        example=torch.load(cache/f"{records[0]['row_id']}.pt",weights_only=True); channels=example['features'][-1].shape[0]
        for name in [m for m in args.models if m.startswith(version)]:
            for seed in args.seeds:
                seed_all(seed); base=SpatialModel(channels,args.width,name.endswith('segdino'),patch=14 if version=='dinov2' else 16); initial=copy.deepcopy(base.state_dict())
                runroot=args.output/'runs'; runroot.mkdir(exist_ok=True)
                preout=runroot/f'{name}__pretrained__{seed}'; preout.mkdir(exist_ok=True)
                do_pre=args.mode=='classification' or 'pretrained' in args.routes
                if do_pre:
                    model=base.to(device)
                    if args.qualitative_only:
                        if not (preout/'classification_best.pt').exists():
                            print(f'Missing classification checkpoint: {preout}; skipping.',flush=True)
                            continue
                        checkpoint=torch.load(preout/'classification_best.pt',map_location='cpu',weights_only=True)
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        train(model,**dict(trainset=datasets['classification']['train'],valset=datasets['classification']['val']),device=device,task='classification',epochs=args.classification_epochs,args=args,out=preout,seed=seed)
                    export_predictions(model,datasets['classification']['val'],device,preout,'classification')
                    export_qualitative(model,datasets['classification']['val'],device,preout,'classification')
                    pretrained=copy.deepcopy({k:v.cpu() for k,v in model.state_dict().items()})
                if args.mode=='classification': continue
                for route in args.routes:
                    out=runroot/f'{name}__{route}__{seed}'; out.mkdir(exist_ok=True)
                    model=SpatialModel(channels,args.width,name.endswith('segdino'),patch=14 if version=='dinov2' else 16)
                    model.load_state_dict(pretrained if route=='pretrained' else initial)
                    # Same fresh segmentation output weights in both routes.
                    model.segmenter.load_state_dict({k.removeprefix('segmenter.'):v for k,v in initial.items() if k.startswith('segmenter.')})
                    model.to(device)
                    if args.qualitative_only:
                        if not (out/'segmentation_best.pt').exists():
                            print(f'Missing segmentation checkpoint: {out}; skipping.',flush=True)
                            continue
                        checkpoint=torch.load(out/'segmentation_best.pt',map_location='cpu',weights_only=True)
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        train(model,**dict(trainset=datasets['segmentation']['train'],valset=datasets['segmentation']['val']),device=device,task='segmentation',epochs=args.segmentation_epochs,args=args,out=out,seed=seed)
                    export_predictions(model,datasets['segmentation']['val'],device,out,'segmentation')
                    # Export all validation scans, including those without annotations.
                    export_qualitative(model,datasets['classification']['val'],device,out,'segmentation',preout/'classification_qualitative' if route=='pretrained' else None)
    qualitative_comparison(args.output,records)
    segmentation_focus_comparison(args.output,records)
    reports(args.output,records,args.size)
    classification=[]
    for path in args.output.glob('runs/*/classification_predictions.csv'):
        df=pd.read_csv(path); classification.append({'experiment':path.parent.name,'balanced_accuracy':balanced_accuracy_score(df.stage,df.prediction),'n_scans':len(df),'n_specimens':df.physical_id.nunique()})
    pd.DataFrame(classification).to_csv(args.output/'classification_comparison.csv',index=False)
    print(f'Finished. Results: {args.output}')


if __name__=='__main__': main()
