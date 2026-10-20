"""Prepared-snapshot loading, independent-specimen protocols and surface controls.
No torch import: audit, planning and data tests can run without the training stack.
"""
from __future__ import annotations
import contextlib, csv, hashlib, json, math, re
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

STAGES=['baseline','partial','clean']
SCANNERS=['LABscanner','TRIOS3','TRIOS5','iTERO']
DEFAULT_ROOT=r'C:\Users\au662213\repos\biocal3d\data\02-09-2026-Data collected\_development_20261013_101905_315169'
VERSION='prepared-investigations-1.0.1'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def signature(x):return hashlib.sha256(json.dumps(x,sort_keys=True,default=str,allow_nan=False).encode()).hexdigest()
def seed_for(x,seed=123):return (int(signature(str(x))[:8],16)+seed)%(2**32)
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2,default=str,allow_nan=False),encoding='utf-8');temp.replace(path)
def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def canonical_scanner(x):
    aliases={'lab':'LABscanner','labscanner':'LABscanner','trios3':'TRIOS3','trios5':'TRIOS5','itero':'iTERO'}
    key=re.sub(r'[^a-z0-9]','',str(x).lower())
    if key not in aliases:raise ValueError(f'Unknown scanner {x!r}; expected {SCANNERS}.')
    return aliases[key]

def identity(text):
    m=re.search(r'BCG(\d+)T(\d+)[_-](\d+)',str(text),re.I)
    if not m:raise ValueError(f'Cannot identify sample BCGxTy-n in {text!r}.')
    g,t,n=map(int,m.groups())
    stage='baseline' if t==0 else ('clean' if g in (1,2) and t==1 else 'partial' if g>=3 and t==1 else 'clean' if g>=3 and t==2 else None)
    if stage is None or g not in range(1,7):raise ValueError(f'Unknown group/timepoint G{g}T{t}.')
    return g,t,n,f'BCG{g}T{t}-{n}',f'BCG{g}-{n}',stage

def normalize_saved_split(value):
    """Preparation's unassigned marker is a placeholder, not a partition."""
    if value is None:return ''
    value=str(value).strip().casefold()
    if value in ('','unassigned'):return ''
    value={'training':'train','validation':'val','testing':'test'}.get(value,value)
    if value not in ('train','val','test'):raise ValueError(f'Unknown saved split {value!r}')
    return value

def saved_split_from_record(row):
    # A placeholder in split must not hide a real assignment in partition.
    parts={normalize_saved_split(row[key]) for key in ('split','partition') if key in row}
    parts.discard('')
    if len(parts)>1:raise ValueError('Conflicting split and partition assignments in snapshot record.')
    return next(iter(parts),'')

def load_snapshot(root,allow_incomplete=False,verify_hashes=True):
    root=Path(root).expanduser().resolve()
    index=root if root.is_file() else root/'dataset_index.json'
    if not index.is_file():raise FileNotFoundError(f'Expected {index}. No fallback to old ROI folders is allowed.')
    root=index.parent;doc=json.loads(index.read_text(encoding='utf-8-sig'))
    source=doc.get('scans') if isinstance(doc,dict) else None
    if not isinstance(source,list):raise ValueError('dataset_index.json must contain a scans list (prepared snapshot schema).')
    rows=[];excluded=[];seen=set();source_ids=set()
    for original in source:
        r=dict(original)
        if str(r.get('status','')).upper()!='OK':
            excluded.append({'scan_id':r.get('scan_id',''),'reason':'status is not OK','status':r.get('status')});continue
        relative=r.get('sample_dir')
        if not relative:raise ValueError(f'Missing sample_dir in {r.get("scan_id",r)}')
        folder=(root/str(relative).replace('\\','/')).resolve()
        if not folder.is_relative_to(root):raise ValueError(f'sample_dir escapes prepared snapshot: {relative}')
        # Identity comes from prepared folder/sample metadata, never from a raw-data fallback.
        names=[r.get('sample_name',''),r.get('sample_id',''),r.get('scan_id',''),relative]
        ids=[identity(s) for s in names if re.search(r'BCG\d+T\d+[_-]\d+',str(s),re.I)]
        if not ids or len({x[3] for x in ids})!=1:raise ValueError(f'Missing or inconsistent sample identity: {r}')
        g,t,n,sample,physical,stage=ids[0]
        if str(r.get('stage','')).lower()!=stage:raise ValueError(f'Stage does not match G/T for {sample}: {r.get("stage")} versus {stage}')
        if 'group' in r and int(re.sub(r'^G','',str(r['group']),flags=re.I))!=g:raise ValueError(f'Group mismatch: {sample}')
        saved_pid=str(r.get('physical_id',''))
        m=re.fullmatch(r'(?:BCG(\d+)[_-](\d+)|G(\d+)_S(\d+))',saved_pid,re.I)
        if not m or tuple(int(x) for x in m.groups() if x is not None)!=(g,n):raise ValueError(f'Invalid physical_id {saved_pid!r} for {sample}')
        scanner=canonical_scanner(r.get('scanner',''));key=(scanner,sample)
        if key in seen:raise ValueError(f'Duplicate scanner/sample: {key}')
        seen.add(key)
        scan_id=str(r.get('scan_id') or scanner+'__'+sample)
        if scan_id in source_ids:raise ValueError(f'Duplicate scan_id {scan_id}')
        source_ids.add(scan_id)
        geometry=folder/'geometry.npz'
        if not geometry.is_file():raise FileNotFoundError(geometry)
        sha=digest(geometry)
        if verify_hashes and r.get('geometry_sha256') and r['geometry_sha256']!=sha:raise ValueError(f'Geometry provenance mismatch: {geometry}')
        row={'row_id':signature([scanner,sample])[:16],'scan_id':scan_id,'scanner':scanner,'sample_name':sample,
             'physical_id':physical,'source_physical_id':saved_pid,'group':g,'timepoint':t,'specimen_number':n,
             'stage':stage,'folder':str(folder),'mesh_path':str(geometry),'geometry_sha256':sha,
             'hash_supplied':bool(r.get('geometry_sha256')),'source_split':saved_split_from_record(r)}
        rows.append(row)
    if not rows:raise ValueError('No OK scans in the snapshot.')
    expected={(s,identity(f'BCG{g}T{t}-{n}')[3]) for s in SCANNERS for g in range(1,7) for n in range(1,9) for t in (range(2) if g<3 else range(3))}
    missing=sorted(expected-seen);extra=sorted(seen-expected)
    audit={'version':VERSION,'index':str(index),'index_sha256':digest(index),'scans':len(rows),
           'physical_specimens':len({r['physical_id'] for r in rows}),'groups':sorted({r['group'] for r in rows}),
           'scanners':sorted({r['scanner'] for r in rows}),'expected_complete_scans':512,
           'missing_scans':[{'scanner':s,'sample_name':n} for s,n in missing],
           'unexpected_scans':[{'scanner':s,'sample_name':n} for s,n in extra],
           'excluded_count':len(excluded),'source_hashes_missing':sum(not r['hash_supplied'] for r in rows)}
    if (missing or extra) and not allow_incomplete:
        raise ValueError(f'Cohort incomplete: {len(rows)}/512 scans; {len(missing)} missing, {len(extra)} unexpected. '
                         f'First missing: {missing[:8]}. Use --allow-incomplete only after reviewing this difference.')
    return sorted(rows,key=lambda r:r['row_id']),audit,excluded

def geometry(row):
    with np.load(row['mesh_path'],allow_pickle=False) as z:
        if not {'vertices','faces'}.issubset(z.files):raise ValueError(f'Expected vertices/faces in {row["mesh_path"]}; keys={z.files}')
        v=np.asarray(z['vertices'],dtype=np.float64);f=np.asarray(z['faces'])
        rgb=None
        for key in ['vertex_rgb','rgb','colors','colours','color']:
            if key in z:
                rgb=np.asarray(z[key],dtype=np.float64);break
    if v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all():raise ValueError(f'Invalid vertices: {row["scan_id"]}')
    if f.ndim!=2 or f.shape[1]!=3 or not np.issubdtype(f.dtype,np.integer):raise ValueError('faces must be integer Nx3 triangles.')
    f=f.astype(np.int64)
    if len(f)<17 or f.min()<0 or f.max()>=len(v):raise ValueError(f'Invalid or insufficient triangles: {row["scan_id"]}')
    if rgb is not None:
        if rgb.ndim!=2 or rgb.shape[0]!=len(v) or rgb.shape[1]<3:raise ValueError('vertex_rgb must be Nx3/Nx4.')
        rgb=rgb[:,:3]
        if not np.isfinite(rgb).all() or rgb.min()<0 or rgb.max()>255:raise ValueError('Invalid RGB range.')
        if rgb.max()>1.001:rgb=rgb/255.
        rgb=np.clip(rgb,0,1)
    tri=v[f];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=np.linalg.norm(cross,axis=1)/2
    good=area>1e-12
    if good.sum()<17:raise ValueError('Fewer than 17 nondegenerate faces.')
    # Retain source face indices; zero-area triangles are never sampled.
    fn=cross/np.maximum(np.linalg.norm(cross,axis=1,keepdims=True),1e-12)
    vn=np.zeros_like(v)
    for k in range(3):np.add.at(vn,f[:,k],cross)
    vn/=np.maximum(np.linalg.norm(vn,axis=1,keepdims=True),1e-12)
    return v,f,vn,rgb,area,fn

def plane_frame(v,f,area):
    centers=v[f].mean(1);center=np.average(centers,axis=0,weights=area)
    points=centers-center;cov=(points.T*area)@points/max(area.sum(),1e-12)
    _,vectors=np.linalg.eigh(cov);normal=vectors[:,0]
    if normal[2]<0:normal=-normal
    x=np.array([1.,0.,0.]);x-=normal*np.dot(x,normal)
    if np.linalg.norm(x)<1e-6:x=np.array([0.,1.,0.]);x-=normal*np.dot(x,normal)
    x/=np.linalg.norm(x);y=np.cross(normal,x)
    return center,np.column_stack([x,y,normal])

def make_surface(row,config,model):
    v,f,vn,rgb,area,fn=geometry(row);center,plane=plane_frame(v,f,area)
    rot=plane if config.get('align_plane',False) else np.eye(3)
    v=(v-center)@rot;vn=vn@rot;fn=fn@rot
    rng=np.random.default_rng(seed_for(row['row_id'],config['sampling_seed']))
    valid=np.flatnonzero(area>1e-12);p=area[valid]/area[valid].sum();count=config['points']
    replace=(len(valid)<count) if config['sampling']=='face_centers' or model=='tsgcnet' else True
    face=rng.choice(valid,count,replace=replace,p=p)
    if config['sampling']=='face_centers' or model=='tsgcnet':bary=np.full((count,3),1/3)
    else:
        u=rng.random((count,2));s=np.sqrt(u[:,0]);bary=np.column_stack([1-s,s*(1-u[:,1]),s*u[:,1]])
    corners=v[f[face]];coord=(corners*bary[:,:,None]).sum(1)/config['scale_mm']
    normal=(vn[f[face]]*bary[:,:,None]).sum(1);normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
    display=np.zeros_like(coord) if rgb is None else (rgb[f[face]]*bary[:,:,None]).sum(1)
    tsgc=np.concatenate([(corners/config['scale_mm']).reshape(-1,9),corners.mean(1)/config['scale_mm'],
                         vn[f[face]].reshape(-1,9),fn[face]],axis=1)
    voxel=config.get('surface_grid_mm',0.)
    if voxel:
        if model=='tsgcnet':raise ValueError('Voxel averaging applies to point encoders, not native TSGCNet face inputs.')
        cell=np.floor(coord*config['scale_mm']/voxel).astype(np.int64)
        _,inv=np.unique(cell,axis=0,return_inverse=True);ncell=int(inv.max())+1;cnt=np.bincount(inv)
        def mean(x):
            out=np.zeros((ncell,x.shape[1]));np.add.at(out,inv,x);return out/cnt[:,None]
        co,no,col=mean(coord),mean(normal),mean(display)
        pick=rng.choice(ncell,count,replace=count>ncell)
        coord,normal,display=co[pick],no[pick],col[pick]
        normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
        # Averaged cells do not have a single original triangle/barycentric map.
        face=np.full(count,-1,dtype=np.int64);bary=np.full((count,3),np.nan)
    else:ncell=count
    color=display.copy() if config['inputs']=='geometry_rgb' else np.zeros_like(display)
    if config.get('xyz_only',False):normal[:]=0;tsgc[:,12:]=0
    data={k:a.astype(np.float32) for k,a in [('coord',coord),('normal',normal),('color',color),('tsgc',tsgc),('display_rgb',display)]}
    data.update(face_index=face.astype(np.int64),barycentric=bary.astype(np.float32),center_mm=center,rotation=rot,
                has_rgb=np.array(rgb is not None and rgb.max()>0),seed=np.array(seed_for(row['row_id'],config['sampling_seed']),dtype=np.int64),
                effective_cells=np.array(ncell),grid_origin=coord.min(0).astype(np.float32))
    return data

def check_surface_rgb(row):
    v,f,n,c,a,fn=geometry(row)
    return c is not None and bool(c.max()>0)

def geometric_features(row):
    v,f,vn,c,area,fn=geometry(row);center,rot=plane_frame(v,f,area);pc=(v[f].mean(1)-center)@rot
    w=area/area.sum();h=pc[:,2];mean=np.sum(w*h);res=h-mean
    features={'height_rq':float(np.sqrt(np.sum(w*res**2))),'height_ra':float(np.sum(w*np.abs(res))),
              'height_p05':float(np.quantile(res,.05)),'height_p95':float(np.quantile(res,.95)),
              'height_range':float(np.ptp(res)),'normal_spread':float(np.sum(w*(1-np.abs(fn@rot[:,2])))),
              'area_mm2':float(area.sum()),'width_mm':float(np.ptp(pc[:,0])),'length_mm':float(np.ptp(pc[:,1]))}
    return features

def color_features(row):
    v,f,n,c,area,fn=geometry(row)
    if c is None or not c.max()>0:return None
    fc=c[f].mean(1);w=area/area.sum();mu=(w[:,None]*fc).sum(0)
    sd=np.sqrt((w[:,None]*(fc-mu)**2).sum(0))
    return {f'rgb_{stat}_{ch}':float(value) for stat,arr in [('mean',mu),('sd',sd)] for ch,value in zip('rgb',arr)}

def assert_partition(rows):
    mapping={}
    for r in rows:mapping.setdefault(r['physical_id'],set()).add(r['split'])
    if any(len(x)>1 for x in mapping.values()):raise ValueError('Physical specimen occurs in multiple partitions.')
    for part in {r['split'] for r in rows}:
        if {r['stage'] for r in rows if r['split']==part}!=set(STAGES):raise ValueError(f'{part} lacks a stage.')

def reference_split(rows,seed=42,explicit=None):
    assignment={}
    saved=[] if explicit else [normalize_saved_split(r.get('source_split')) for r in rows]
    if explicit:
        doc=json.loads(Path(explicit).read_text(encoding='utf-8-sig'))
        for part,ids in doc['physical_ids'].items():
            if part not in ['train','val','test']:raise ValueError('Unknown partition in split JSON.')
            for pid in ids:
                m=re.fullmatch(r'(?:BCG(\d+)[_-](\d+)|G(\d+)_S(\d+))',pid,re.I)
                if not m:raise ValueError(f'Unknown specimen identity {pid}')
                g,n=[int(x) for x in m.groups() if x is not None];pid=f'BCG{g}-{n}'
                if pid in assignment:raise ValueError('Split JSON has duplicate specimen IDs.')
                assignment[pid]=part
        if set(assignment)!={r['physical_id'] for r in rows}:raise ValueError('Split JSON must cover exactly this cohort.')
    elif all(saved):
        for r,part in zip(rows,saved):
            if r['physical_id'] in assignment and assignment[r['physical_id']]!=part:raise ValueError('Saved snapshot split leaks specimens.')
            assignment[r['physical_id']]=part
    elif any(saved):raise ValueError('Snapshot provides real split assignments for only some scans; other scans are unassigned. Supply a complete --split-json to resolve this without overwriting existing assignments.')
    else:
        for g in sorted({r['group'] for r in rows}):
            p=sorted({r['physical_id'] for r in rows if r['group']==g})
            np.random.default_rng(seed_for(g,seed)).shuffle(p)
            if len(p)<4:raise ValueError('Need at least four specimens per group for reference train/val/test.')
            n=max(1,round(len(p)*.125))
            for i,pid in enumerate(p):assignment[pid]='test' if i<n else 'val' if i<2*n else 'train'
    result=[dict(r,split=assignment[r['physical_id']]) for r in rows];assert_partition(result)
    return result

def assessment_folds(reference,folds,seed):
    # Reserve global test subjects. Outer assessment subjects never select checkpoints.
    dev=[r for r in reference if r['split']!='test']
    meta={r['physical_id']:r['group'] for r in dev};ids=sorted(meta);groups=np.array([meta[p] for p in ids])
    if min(list(groups).count(g) for g in set(groups))<folds:raise ValueError('Not enough development specimens per group for requested folds.')
    skf=StratifiedKFold(folds,shuffle=True,random_state=seed)
    for k,(train_idx,assess_idx) in enumerate(skf.split(ids,groups)):
        assigned={ids[i]:'assessment' for i in assess_idx}
        for g in sorted(set(groups)):
            pool=[ids[i] for i in train_idx if groups[i]==g]
            np.random.default_rng(seed_for(f'{g}:{k}',seed)).shuffle(pool)
            if len(pool)<2:raise ValueError('Outer training pool cannot support separate inner validation.')
            for j,p in enumerate(pool):assigned[p]='val' if j==0 else 'train'
        rows=[dict(r,split=assigned[r['physical_id']]) for r in dev];assert_partition(rows)
        yield k,rows

def loso_rows(rows,scanner):
    selected=[r for r in rows if (r['split']=='assessment' and r['scanner']==scanner) or (r['split']!='assessment' and r['scanner']!=scanner)]
    assert_partition(selected)
    if any(r['scanner']==scanner for r in selected if r['split'] in ['train','val']):raise AssertionError('Scanner leakage.')
    return selected

def score(rows,prob):
    truth=np.array([STAGES.index(r['stage']) for r in rows]);pred=prob.argmax(1)
    cm=confusion_matrix(truth,pred,labels=[0,1,2]);support=cm.sum(1)
    rec=np.divide(np.diag(cm),support,out=np.zeros(3,dtype=float),where=support>0)
    perclass=[]
    for c in range(3):
        vals=[]
        for p in sorted({r['physical_id'] for r in rows}):
            mask=np.array([r['physical_id']==p for r in rows])&(truth==c)
            if mask.any():vals.append(float((pred[mask]==c).mean()))
        perclass.append(float(np.mean(vals)) if vals else None)
    return {'accuracy':float((truth==pred).mean()),'balanced_accuracy':float(rec.mean()) if (support>0).all() else None,
            'specimen_balanced_accuracy':float(np.mean(perclass)) if all(x is not None for x in perclass) else None,
            'macro_f1':float(f1_score(truth,pred,labels=[0,1,2],average='macro',zero_division=0)),
            'macro_auc':float(roc_auc_score(truth,prob,multi_class='ovr',average='macro')) if (support>0).all() else None,
            'recall':rec.tolist(),'class_support':support.tolist(),'confusion_matrix':cm.tolist(),
            'scans':len(rows),'specimens':len({r['physical_id'] for r in rows}),
            'partial_specimens':len({r['physical_id'] for r in rows if r['stage']=='partial'}),
            'loss':float(-np.log(np.maximum(prob[np.arange(len(rows)),truth],1e-12)).mean())}

def cluster_interval(rows,prob,repeats=500,seed=123):
    # Stratify by stage availability, not each G group: with one val subject per G,
    # a per-G bootstrap would be degenerate. Retain all scans/stages per specimen.
    groups={g:sorted({r['physical_id'] for r in rows if ('fresh' if r['group']<3 else 'mature')==g}) for g in ['fresh','mature']}
    groups={g:p for g,p in groups.items() if p}
    support={c:len({r['physical_id'] for r in rows if r['stage']==c}) for c in STAGES}
    if min(support.values())<3:return {'interval':None,'reason':'Fewer than three independent specimens in at least one stage.','specimens_per_stage':support}
    rng=np.random.default_rng(seed);values=[]
    for _ in range(repeats):
        sampled=[];pp=[];j=0
        for group,pids in groups.items():
            for pid in rng.choice(pids,len(pids),replace=True):
                ix=[i for i,r in enumerate(rows) if r['physical_id']==pid]
                sampled.extend(dict(rows[i],physical_id=f'bootstrap-{j}') for i in ix);pp.extend(prob[ix]);j+=1
        val=score(sampled,np.asarray(pp))['specimen_balanced_accuracy']
        if val is not None:values.append(val)
    return {'interval':[float(v) for v in np.quantile(values,[.025,.975])],
            'method':'95% percentile; fresh/mature-stratified whole-specimen bootstrap; fixed fitted model',
            'repeats':repeats,'seed':seed,'specimens_per_stage':support,
            'limitation':'Conditional on this fit/split; does not capture retraining, seed or model-selection variability.'}

def transformed(data,kind):
    d={k:v.copy() for k,v in data.items()}
    if kind=='rotate_z_90':
        rot=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        d['coord']=d['coord']@rot.T;d['normal']=d['normal']@rot.T
        for a,b in [(0,9),(9,12),(12,21),(21,24)]:d['tsgc'][:,a:b]=(d['tsgc'][:,a:b].reshape(-1,3)@rot.T).reshape(-1,b-a)
        d['grid_origin']=d['coord'].min(0)
    elif kind=='translate_scaled_0p1':
        shift=np.array([.1,-.1,.05]);d['coord']+=shift;d['tsgc'][:,:9]+=np.tile(shift,3);d['tsgc'][:,9:12]+=shift
        d['grid_origin']+=shift
    else:raise ValueError(kind)
    return d
