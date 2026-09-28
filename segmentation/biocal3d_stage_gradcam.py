"""Train a stage-classification head on frozen spatial DINOv2 features.

Place beside biocal3d_prepare_dino.py. Requires torch, numpy, pandas,
scikit-learn, pillow, matplotlib. First regenerate maps and DINO features with
updated preparation script (including map hashes).

python biocal3d_stage_gradcam.py --live
python biocal3d_stage_gradcam.py --step

Image-level labels: baseline / partial / clean. These are known experimental
stages used as WEAK supervision for plaque localization, not pixel labels.
Encoder remains frozen. Only a small spatial classification head is trained.
Grad-CAM is computed at that head's hidden spatial feature layer, not attention.
No test-set inference. Best epoch selected by validation cross entropy.
With 4 validation specimens, results are exploratory and uncertain.
"""
from pathlib import Path
import argparse
import hashlib
import json
import random
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from biocal3d_prepare_dino import DATA_ROOT, LivePreview, save_preview

CLASSES = ['baseline', 'partial', 'clean']


class Features(Dataset):
    def __init__(self, rows):
        self.rows = rows.reset_index(drop=True)
        self.items = []
        for row in self.rows.to_dict('records'):
            folder = Path(row['map_dir'])
            with np.load(folder/'dino_features.npz') as z:
                digest = hashlib.sha256((folder/'maps.npz').read_bytes()).hexdigest()
                if 'map_sha256' not in z or str(z['map_sha256']) != digest:
                    raise ValueError(f'Stale or unverified DINO features: {folder}. Rerun preparation --feature-only.')
                f = z['features'].astype(np.float32).transpose(2,0,1)
                mask = z['patch_valid'].astype(np.float32)[None]
            if mask.sum() == 0:
                raise ValueError(f'No valid feature patches: {folder}')
            self.items.append((torch.from_numpy(f), torch.from_numpy(mask), CLASSES.index(row['stage'])))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class StageHead(nn.Module):
    def __init__(self, channels, hidden=64):
        super().__init__()
        self.spatial = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.ReLU())
        self.classifier = nn.Conv2d(hidden, len(CLASSES), 1)

    def forward(self, x, mask, return_features=False):
        activation = self.spatial(x)
        spatial_logits = self.classifier(activation)
        logits = (spatial_logits*mask).sum((2,3))/mask.sum((2,3)).clamp_min(1)
        return (logits, activation) if return_features else logits


def gradcam(model, x, mask, target):
    # Frozen encoder features are constants; head parameters create autograd graph.
    model.zero_grad(set_to_none=True)
    logits, activation = model(x, mask, return_features=True)
    gradient = torch.autograd.grad(logits[:, target].sum(), activation)[0]
    weights = (gradient*mask).sum((2,3),keepdim=True)/mask.sum((2,3),keepdim=True).clamp_min(1)
    cam = torch.relu((weights*activation).sum(1,keepdim=True))*mask
    raw_max = cam.amax().item()
    cam = cam/cam.amax((2,3),keepdim=True).clamp_min(1e-12)
    return cam.detach()[0,0].cpu().numpy(), raw_max


def evaluate(model, loader, device):
    ys, preds, probabilities = [], [], []
    loss_sum = 0.
    model.eval()
    with torch.no_grad():
        for x, mask, y in loader:
            x,mask,y=x.to(device),mask.to(device),y.to(device)
            logits=model(x,mask)
            loss_sum += nn.functional.cross_entropy(logits,y,reduction='sum').item()
            ys.extend(y.cpu().tolist());preds.extend(logits.argmax(1).cpu().tolist())
            probabilities.extend(logits.softmax(1).cpu().tolist())
    return loss_sum/len(ys), balanced_accuracy_score(ys,preds), ys, preds, probabilities


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root',type=Path,default=DATA_ROOT/'_dino_preparation_80_10_10')
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--patience',type=int,default=15)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--device',default='auto')
    p.add_argument('--live',action='store_true')
    p.add_argument('--step',action='store_true')
    args=p.parse_args()
    if min(args.epochs,args.patience,args.batch_size)<=0 or args.lr<=0:
        p.error('Epochs, patience, batch size and learning rate must be positive')
    torch.manual_seed(42);np.random.seed(42);random.seed(42)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    manifest=pd.read_csv(args.root/'scan_manifest.csv')
    if manifest.groupby('physical_id').split.nunique().max()!=1:
        raise ValueError('Specimen leakage across splits')
    trainrows=manifest[manifest.split=='train'].copy()
    valrows=manifest[manifest.split=='val'].copy()
    for name, rows in [('train',trainrows),('val',valrows)]:
        if set(rows.stage)!=set(CLASSES):
            raise ValueError(f'{name} must contain all three stages; inspect missing scans before proceeding')
    out=args.root/'stage_classification'
    out.mkdir(exist_ok=True)
    train=Features(trainrows);val=Features(valrows)
    trainloader=DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=0)
    valloader=DataLoader(val,batch_size=args.batch_size,shuffle=False,num_workers=0)
    channels=train[0][0].shape[0]
    model=StageHead(channels).to(device)
    counts=np.bincount([item[2] for item in train.items],minlength=3)
    weights=torch.tensor(len(train)/(3*counts),dtype=torch.float32,device=device)
    criterion=nn.CrossEntropyLoss(weight=weights)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-3)
    best=float('inf');best_epoch=0;state=None;history=[]
    print(f'Train: {len(train)} scans / {trainrows.physical_id.nunique()} specimens. '
          f'Validation: {len(val)} scans / {valrows.physical_id.nunique()} specimens. Device: {device}',flush=True)
    print('Training head only. Encoder frozen. Stages are not pixel-level plaque labels.',flush=True)
    for epoch in range(1,args.epochs+1):
        model.train();running=0
        for x,mask,y in trainloader:
            x,mask,y=x.to(device),mask.to(device),y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(x,mask),y);loss.backward();optimizer.step()
            running+=loss.item()*len(y)
        vl,ba,_,_,_=evaluate(model,valloader,device)
        history.append(dict(epoch=epoch,weighted_train_loss=running/len(train),val_loss=vl,val_balanced_accuracy=ba))
        pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
        print(f'Epoch {epoch:03d}: train loss={running/len(train):.4f}, val loss={vl:.4f}, val balanced accuracy={ba:.3f}',flush=True)
        if vl<best:
            best=vl;best_epoch=epoch;state=copy.deepcopy(model.state_dict())
            torch.save(dict(state_dict=state,channels=channels,classes=CLASSES,best_epoch=epoch),out/'best_stage_head.pt')
        if epoch-best_epoch>=args.patience:
            print('Early stopping.',flush=True);break
    model.load_state_dict(state);model.eval()
    viewer=LivePreview(args.live or args.step,args.step)
    allpred=[]
    try:
        for split,ds in [('train',train),('val',val)]:
            loss,ba,ys,preds,probs=evaluate(model,DataLoader(ds,batch_size=args.batch_size),device)
            pd.DataFrame(confusion_matrix(ys,preds,labels=[0,1,2]),index=CLASSES,columns=CLASSES).to_csv(out/f'{split}_confusion_matrix.csv')
            for i,row in enumerate(ds.rows.to_dict('records')):
                x,mask,y=ds[i];x=x[None].to(device);mask=mask[None].to(device)
                folder=out/'gradcam'/split/row['scanner']/row['sample_name'];folder.mkdir(parents=True,exist_ok=True)
                with np.load(Path(row['map_dir'])/'maps.npz') as z:
                    rgb=z['rgb'];valid=z['valid']
                from PIL import Image
                from matplotlib import colormaps
                overlays=[];record={**row,'predicted_stage':CLASSES[preds[i]]}
                for c,name in enumerate(CLASSES):
                    cam,raw_max=gradcam(model,x,mask,c)
                    full = np.array(
                        Image.fromarray(cam).resize(
                            (rgb.shape[1], rgb.shape[0]),
                            Image.Resampling.BILINEAR,
                        ),
                        dtype=np.float32,
                        copy=True,
                    )
                    full[~valid]=0
                    heat=colormaps['inferno'](full)[...,:3]
                    overlay=rgb.copy();overlay[valid]=.6*rgb[valid]+.4*heat[valid]
                    overlays.append(overlay)
                    np.savez_compressed(folder/f'gradcam_{name}.npz',normalized_cam=full,raw_positive_max=raw_max)
                    Image.fromarray(np.uint8(np.clip(overlay,0,1)*255)).save(folder/f'gradcam_{name}.png')
                    record[f'p_{name}']=probs[i][c]
                    record[f'cam_positive_max_{name}']=raw_max
                caption=f"{row['scanner']} {row['sample_name']} | stage={row['stage']} predicted={CLASSES[preds[i]]}"
                titles=['RGB','Baseline-class Grad-CAM','Clean-class Grad-CAM']
                save_preview(folder,rgb,overlays[0],overlays[2],caption,titles,'comparison.png')
                print(f'[Grad-CAM {split} {i+1}/{len(ds)}] {caption}',flush=True)
                viewer.show([rgb,overlays[0],overlays[2]],titles,caption)
                allpred.append(record)
        predictions=pd.DataFrame(allpred)
        predictions.to_csv(out/'stage_predictions.csv',index=False)
        metrics=[]
        for keys,group in predictions.groupby(['split','scanner']):
            metrics.append(dict(split=keys[0],scanner=keys[1],n_scans=len(group),
                balanced_accuracy=balanced_accuracy_score(group.stage,group.predicted_stage)))
        pd.DataFrame(metrics).to_csv(out/'metrics_by_scanner.csv',index=False)
        (out/'run.json').write_text(json.dumps(dict(classes=CLASSES,best_epoch=best_epoch,
            best_val_loss=best,encoder='frozen cached dinov2_vits14_reg',
            target_layer='StageHead.spatial ReLU output',device=device,
            interpretation='Stage attribution, not plaque segmentation. CAM normalized per image/class; intensity is not confidence.',
            test_used=False),indent=2))
    finally:
        viewer.close()
    print(f'Done: {out}',flush=True)

if __name__=='__main__':
    main()
