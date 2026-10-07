"""Train and compare BioCal3D face and point classifiers in an existing environment."""
from pathlib import Path
import argparse, hashlib, json, platform, random, shutil, sys, time
import numpy as np
import torch
from data3d import (load_index, make_splits, apply_splits, validate_geometry, inputs, to_device,
                    save_json, write_csv, digest)
from models3d import build
from report3d import metrics, plot_run, compare


def atomic_torch_save(obj,path):
    tmp=path.with_name(path.name+'.tmp'); torch.save(obj,tmp); tmp.replace(path)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def evaluate(net,rows,cfg,device):
    net.eval(); probabilities=[]
    with torch.no_grad():
        for r in rows:
            batch,_=inputs(r,cfg['model'],cfg['elements'],cfg['colour'],cfg['scale_mm'],cfg['sampling_seed'])
            probabilities.append(net(to_device(batch,device)).softmax(-1)[0].cpu().numpy())
    p=np.array(probabilities); y=np.array([r['target'] for r in rows])
    return metrics(y,p,cfg['classes']),p


def predictions(path,rows,probs,classes):
    out=[]
    for r,p in zip(rows,probs):
        out.append(dict(scan_id=r['scan_id'],physical_id=r['physical_id'],scanner=r['scanner'],
            group=r['group'],growth_condition=r.get('growth_condition',''),target=r['target'],label=r['label'],
            predicted=classes[int(p.argmax())],**{'p_'+c:float(v) for c,v in zip(classes,p)}))
    write_csv(path,out)


def train_one(rows,classes,output,cfg,args,device):
    folder=output/f"{cfg['model']}_{cfg['input_mode']}_seed{cfg['seed']}"
    resumed = folder.exists()
    if resumed:
        if not args.resume: raise ValueError(f'Run exists: {folder}. Use --resume with identical settings, or a new --output.')
        previous=json.loads((folder/'config.json').read_text())
        keys=['model','input_mode','classes','seed','sampling_seed','elements','scale_mm','radius','index_hash','split_hash','targets_hash']
        training_keys=['epochs','patience','batch_size','lr','weight_decay','no_augment','exclude_train_scanners','amp']
        if any(previous[k]!=cfg[k] for k in keys) or any(previous['args'][k]!=cfg['args'][k] for k in training_keys):
            raise ValueError('Resume settings differ from saved run; use a new output for a new experiment.')
        cfg=previous
        if (folder/'finished.json').exists():
            print('Reusing completed run:',folder.name,flush=True); return folder
    else:
        folder.mkdir(parents=True); save_json(folder/'config.json',cfg)
    set_seed(cfg['seed']); net=build(cfg['model'],len(classes),cfg['colour'],radius=cfg['radius']).to(device)
    cfg['parameters']=sum(p.numel() for p in net.parameters()); save_json(folder/'config.json',cfg)
    train=[r for r in rows if r['partition']=='train']; val=[r for r in rows if r['partition']=='val']; test=[r for r in rows if r['partition']=='test']
    if args.exclude_train_scanners:
        train=[r for r in train if r['scanner'] not in args.exclude_train_scanners]
        val=[r for r in val if r['scanner'] not in args.exclude_train_scanners]
    if set(r['target'] for r in train)!=set(range(len(classes))) or set(r['target'] for r in val)!=set(range(len(classes))):
        raise ValueError('Scanner exclusion removed an entire train/validation class.')
    counts=np.bincount([r['target'] for r in train],minlength=len(classes)); weights=counts.sum()/(len(classes)*counts)
    criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device))
    opt=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs,eta_min=args.lr*.02)
    amp=args.amp and device.type=='cuda'; scaler=torch.amp.GradScaler('cuda',enabled=amp)
    history=[]; best=(-1.,-1.,-float('inf')); stale=0; first_epoch=1; elapsed=0.
    if resumed and (folder/'last.pt').exists():
        last=torch.load(folder/'last.pt',map_location=device,weights_only=False)
        net.load_state_dict(last['state_dict']); opt.load_state_dict(last['optimizer']); scheduler.load_state_dict(last['scheduler'])
        scaler.load_state_dict(last['scaler']); first_epoch=last['epoch']+1; best=tuple(last['best_key']); stale=last['stale']
        history=last['history']; elapsed=last['elapsed_seconds']
        torch.set_rng_state(last['torch_rng'].cpu())
        if device.type=='cuda' and last['cuda_rng'] is not None: torch.cuda.set_rng_state_all([s.cpu() for s in last['cuda_rng']])
        save_json(folder/'history.json',history)
    started=time.perf_counter()-elapsed
    for epoch in range(first_epoch,args.epochs+1):
        if stale>=args.patience: break
        net.train(); order=np.random.default_rng(cfg['seed']+epoch).permutation(len(train)); loss_total=0
        for start in range(0,len(order),args.batch_size):
            chunk=order[start:start+args.batch_size]; opt.zero_grad(set_to_none=True)
            if cfg['model']=='pointnext_s':
                items=[inputs(train[ix],cfg['model'],cfg['elements'],cfg['colour'],cfg['scale_mm'],
                    cfg['sampling_seed'],epoch,augment=not args.no_augment)[0] for ix in chunk]
                batch={k:torch.cat([b[k] for b in items],0) for k in items[0]}
                with torch.autocast(device_type=device.type,enabled=amp):
                    logits=net(to_device(batch,device)); target=torch.tensor([train[ix]['target'] for ix in chunk],device=device)
                    loss=torch.nn.functional.cross_entropy(logits,target,weight=criterion.weight,reduction='sum')/len(chunk)
                if not torch.isfinite(loss): raise ValueError('Nonfinite training loss; check input units and learning rate.')
                scaler.scale(loss).backward(); loss_total+=float(loss.detach())*len(chunk)
            else:
                for ix in chunk:
                    r=train[ix]; batch,_=inputs(r,cfg['model'],cfg['elements'],cfg['colour'],cfg['scale_mm'],cfg['sampling_seed'],epoch,augment=not args.no_augment)
                    with torch.autocast(device_type=device.type,enabled=amp):
                        logits=net(to_device(batch,device)); target=torch.tensor([r['target']],device=device)
                        # Single-item mean CE would cancel weights. Use sum before accumulation.
                        loss=torch.nn.functional.cross_entropy(logits,target,weight=criterion.weight,reduction='sum')/len(chunk)
                    if not torch.isfinite(loss): raise ValueError('Nonfinite training loss; check input units and learning rate.')
                    scaler.scale(loss).backward(); loss_total+=float(loss.detach())*len(chunk)
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            scaler.step(opt); scaler.update()
        scheduler.step()
        tr,_=evaluate(net,train,cfg,device); va,p= evaluate(net,val,cfg,device)
        rec=dict(epoch=epoch,train=tr,val=va,weighted_training_loss=loss_total/len(train),lr=opt.param_groups[0]['lr'],elapsed_seconds=time.perf_counter()-started)
        history.append(rec); save_json(folder/'history.json',history)
        key=(va['balanced_accuracy'],va['macro_f1'],-va['cross_entropy'])
        if key>best:
            best=key; stale=0
            atomic_torch_save(dict(schema_version=1,config=cfg,state_dict=net.state_dict(),epoch=epoch,validation=va),folder/'best.pt')
            atomic_torch_save(dict(config=cfg,state_dict=net.encoder_state(),epoch=epoch),folder/'encoder.pt')
            predictions(folder/'val_predictions.csv',val,p,classes)
        else: stale+=1
        atomic_torch_save(dict(config=cfg,state_dict=net.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(),epoch=epoch,best_key=best,stale=stale,history=history,
            elapsed_seconds=time.perf_counter()-started,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None),folder/'last.pt')
        print(f"{folder.name} epoch {epoch}/{args.epochs}: train BA={tr['balanced_accuracy']:.3f}, val BA={va['balanced_accuracy']:.3f}, val F1={va['macro_f1']:.3f}",flush=True)
        if stale>=args.patience: break
    saved=torch.load(folder/'best.pt',map_location=device,weights_only=False); net.load_state_dict(saved['state_dict'])
    score,p=evaluate(net,test,cfg,device)
    save_json(folder/'test_metrics.json',dict(**score,best_epoch=saved['epoch'],parameters=cfg['parameters'],training_seconds=time.perf_counter()-started))
    predictions(folder/'test_predictions.csv',test,p,classes); plot_run(folder,classes)
    save_json(folder/'finished.json',dict(best_epoch=saved['epoch'],completed=True))
    return folder


def prepare(args):
    rows,classes,excluded=load_index(args.index,args.labels_csv,args.classes)
    output=Path(args.output).resolve(); output.mkdir(parents=True,exist_ok=True)
    splits_path=output/'splits.json'
    provenance=output/'provenance.json'
    if provenance.exists() and json.loads(provenance.read_text())['index_hash'] != digest(args.index):
        raise ValueError('Existing output belongs to another snapshot. Use a new --output.')
    if args.splits: split=json.loads(Path(args.splits).read_text(encoding='utf-8-sig'))
    elif splits_path.exists(): split=json.loads(splits_path.read_text())
    else: split=make_splits(rows,args.split_seed,args.fold,args.folds)
    apply_splits(rows,split,classes)
    targets_hash=hashlib.sha256(json.dumps([(r['scan_id'],r['label']) for r in rows],sort_keys=True).encode()).hexdigest()
    if provenance.exists() and json.loads(provenance.read_text())['targets_hash']!=targets_hash:
        raise ValueError('Existing output uses different labels. Use a new --output.')
    if splits_path.exists() and any(output.glob('*/config.json')) and json.loads(splits_path.read_text()) != split:
        raise ValueError('Existing output has trained runs with another split. Use a new --output.')
    save_json(splits_path,split)
    assignment=[{k:r.get(k,'') for k in ['scan_id','physical_id','scanner','group','stage','label','partition']} for r in rows]
    write_csv(output/'split_assignments.csv',assignment); write_csv(output/'excluded_scans.csv',excluded)
    counts={part:{c:sum(r['partition']==part and r['label']==c for r in rows) for c in classes} for part in ['train','val','test']}
    print('Shared splits:',json.dumps(counts),flush=True)
    save_json(output/'data_summary.json',dict(classes=classes,counts=counts,
        specimens={p:len({r['physical_id'] for r in rows if r['partition']==p}) for p in counts},
        scanners=sorted({r['scanner'] for r in rows}),excluded=len(excluded)))
    return rows,classes,output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('check-env')
    for name in ['run','train','make-splits']:
        p=commands.add_parser(name)
        p.add_argument('--index',type=Path,required=True,help='Prepared snapshot dataset_index.json')
        p.add_argument('--output',type=Path,required=True)
        p.add_argument('--labels-csv',type=Path,help='Optional scan_id,label CSV; unmatched scans excluded')
        p.add_argument('--classes',nargs='+',default=['baseline','partial','clean'])
        p.add_argument('--splits',type=Path,help='Existing physical_ids split JSON, reused across models/new scanners')
        p.add_argument('--split-seed',type=int,default=20261008)
        p.add_argument('--fold',type=int,help='Optional zero-based test fold, with next fold for validation')
        p.add_argument('--folds',type=int,default=5)
        if name=='make-splits': continue
        p.add_argument('--models',nargs='+',choices=['tsgcnet','pointnext_s'],default=['tsgcnet','pointnext_s'])
        p.add_argument('--inputs',nargs='+',choices=['geometry','geometry_rgb'],default=['geometry','geometry_rgb'])
        p.add_argument('--seeds',nargs='+',type=int,default=[42])
        p.add_argument('--sampling-seed',type=int,default=1234)
        p.add_argument('--epochs',type=int,default=100); p.add_argument('--patience',type=int,default=20)
        p.add_argument('--batch-size',type=int,default=4,help='PointNeXt batch size; TSGCNet gradient accumulation across variable-size meshes')
        p.add_argument('--faces',type=int,default=1024); p.add_argument('--points',type=int,default=1024)
        p.add_argument('--scale-mm',type=float,default=6.7,help='Fixed spatial divisor for every scan')
        p.add_argument('--radius',type=float,default=.15,help='PointNeXt ball radius in fixed scaled coordinates')
        p.add_argument('--lr',type=float,default=.001); p.add_argument('--weight-decay',type=float,default=.01)
        p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
        p.add_argument('--threads',type=int,default=4); p.add_argument('--amp',action='store_true')
        p.add_argument('--no-augment',action='store_true')
        p.add_argument('--resume',action='store_true',help='Resume interrupted runs / reuse completed runs with identical settings')
        p.add_argument('--exclude-train-scanners',nargs='*',default=[],help='Exclude scanners from training/validation, retain in test')
        p.add_argument('--bootstrap',type=int,default=1000)
        p.add_argument('--explain-count',type=int,default=6,help='Run only: number of shared test scans to explain per trained model')
    p=commands.add_parser('compare'); p.add_argument('--output',type=Path,required=True); p.add_argument('--bootstrap',type=int,default=1000)
    p=commands.add_parser('explain')
    p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--index',type=Path,required=True)
    p.add_argument('--output',type=Path); p.add_argument('--scan-ids',nargs='+'); p.add_argument('--count',type=int,default=6)
    p.add_argument('--target',default='predicted',help='predicted, true, or an explicit class name')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto'); p.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    if args.command=='check-env':
        import scipy,sklearn,matplotlib,plotly
        print('Python:',sys.executable,'\nTorch:',torch.__version__,'CUDA available:',torch.cuda.is_available())
        print('Other versions:',scipy.__version__,sklearn.__version__,matplotlib.__version__,plotly.__version__)
        if torch.cuda.is_available(): print('GPU:',torch.cuda.get_device_name(0))
        return
    if args.command=='compare': compare(args.output,args.bootstrap); return
    torch.set_num_threads(args.threads if hasattr(args,'threads') else 4)
    if args.command=='explain':
        from explain3d import explain_checkpoint
        explain_checkpoint(args.checkpoint,args.index,args.output,args.scan_ids,args.count,args.target,args.device); return
    rows,classes,output=prepare(args)
    if args.command=='make-splits': return
    for key in ['epochs','patience','batch_size','threads']:
        if getattr(args,key)<1: raise ValueError(key+' must be positive.')
    if args.faces<32 or args.points<32 or args.scale_mm<=0 or args.radius<=0: raise ValueError('Use >=32 faces/points and positive scale/radius.')
    if len(set(classes))!=len(classes) or len(classes)<2: raise ValueError('At least two distinct classes required.')
    validate_geometry(rows,'geometry_rgb' in args.inputs)
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device))
    if device.type=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA unavailable. See README installation instructions; CPU also supported.')
    index_hash=digest(args.index); split_hash=digest(output/'splits.json')
    targets_hash=hashlib.sha256(json.dumps([(r['scan_id'],r['label']) for r in rows],sort_keys=True).encode()).hexdigest()
    labels_copy=None
    if args.labels_csv:
        labels_copy=output/'labels.csv'
        if args.labels_csv.resolve()!=labels_copy: shutil.copyfile(args.labels_csv,labels_copy)
    provenance=output/'provenance.json'
    if provenance.exists() and json.loads(provenance.read_text())['index_hash']!=index_hash:
        raise ValueError('Existing output belongs to another snapshot. Use a new --output.')
    code_hashes={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
    if provenance.exists() and json.loads(provenance.read_text())['code_hashes']!=code_hashes:
        raise ValueError('Code changed within an existing experiment. Use a new --output.')
    save_json(provenance,dict(index_hash=index_hash,index=str(args.index.resolve()),split_hash=split_hash,targets_hash=targets_hash,
        versions=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__),
        code_hashes=code_hashes))
    trained=[]
    for seed in args.seeds:
        for mode in args.inputs:
            for name in args.models:
                cfg=dict(model=name,input_mode=mode,colour=mode=='geometry_rgb',classes=classes,seed=seed,sampling_seed=args.sampling_seed,
                    elements=args.faces if name=='tsgcnet' else args.points,scale_mm=args.scale_mm,radius=args.radius,
                    index_hash=index_hash,split_hash=split_hash,targets_hash=targets_hash,index=str(args.index.resolve()),
                    labels_csv=str(labels_copy) if labels_copy else None,
                    args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},device=str(device))
                trained.append(train_one(rows,classes,output,cfg,args,device))
    compare(output,args.bootstrap)
    if args.command=='run' and args.explain_count>0:
        from explain3d import explain_checkpoint, select_scans
        selected=select_scans([r for r in rows if r['partition']=='test'],args.explain_count)
        for folder in trained:
            explain_checkpoint(folder/'best.pt',args.index,None,[r['scan_id'] for r in selected],args.explain_count,'predicted',str(device))
    print('Finished. Open:',output/'comparison.html',flush=True)

if __name__=='__main__':
    try: main()
    except (ValueError,FileNotFoundError,KeyError) as e:
        print('ERROR:',e,file=sys.stderr); sys.exit(2)
