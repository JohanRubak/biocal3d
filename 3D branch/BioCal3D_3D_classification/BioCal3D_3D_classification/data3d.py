"""BioCal3D snapshot loading, specimen splits and deterministic surface sampling."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
import torch

CLASSES = ['baseline', 'partial', 'clean']

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def seed_for(text, seed):
    return (int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) + seed) % (2**32)

def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    path=Path(path); tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8'); tmp.replace(path)

def write_csv(path, rows):
    if not rows: return
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def load_index(index, labels_csv=None, classes=None):
    index = Path(index).resolve()
    doc = json.loads(index.read_text(encoding='utf-8-sig'))
    classes = classes or CLASSES
    if len(classes)<2 or len(set(classes))!=len(classes): raise ValueError('At least two distinct class names required.')
    overrides = None
    if labels_csv:
        items = list(csv.DictReader(open(labels_csv, encoding='utf-8-sig')))
        overrides = {r['scan_id']: r['label'] for r in items}
        if len(overrides) != len(items): raise ValueError('Duplicate scan_id in labels CSV.')
    rows, excluded = [], []
    for r0 in doc['scans']:
        r = dict(r0)
        r['label'] = overrides.get(r['scan_id'], '') if overrides is not None else r.get('stage', '')
        if r.get('status') != 'OK' or r['label'] not in classes:
            excluded.append(dict(scan_id=r['scan_id'], reason='status or unrecognised/missing label', label=r['label']))
            continue
        if not r.get('physical_id'): raise ValueError('Missing physical_id: ' + r['scan_id'])
        relative = Path(r['sample_dir'].replace('\\', '/'))
        folder = (index.parent / relative).resolve()
        if not folder.is_relative_to(index.parent): raise ValueError('sample_dir escapes snapshot.')
        r['folder'] = str(folder); r['target'] = classes.index(r['label'])
        rows.append(r)
    if not rows: raise ValueError('No eligible scans in the index.')
    if len({r['scan_id'] for r in rows}) != len(rows): raise ValueError('Duplicate scan IDs.')
    return rows, classes, excluded

def make_splits(rows, seed, fold=None, folds=5):
    """Stratify by experimental group; all scanner/time replicas stay together."""
    groups, meta = {}, {}
    for r in rows:
        p = r['physical_id']; g = str(r['group'])
        if p in meta and meta[p] != g: raise ValueError('Inconsistent specimen group.')
        meta[p] = g; groups.setdefault(g, set()).add(p)
    parts = {x: [] for x in ['train', 'val', 'test']}
    for group, ps in sorted(groups.items()):
        ps = sorted(ps); np.random.default_rng(seed_for(group, seed)).shuffle(ps)
        if len(ps) < 3: raise ValueError(f'Group {group} needs at least 3 physical specimens for disjoint splits.')
        if fold is None:
            n = max(1, round(len(ps)*.15))
            parts['test'] += ps[:n]; parts['val'] += ps[n:2*n]; parts['train'] += ps[2*n:]
        else:
            if folds < 3 or not 0 <= fold < folds or len(ps) < folds:
                raise ValueError('For folds: >=3 folds, valid fold index, and >=folds specimens per group required.')
            for i, p in enumerate(ps):
                part = 'test' if i % folds == fold else ('val' if i % folds == (fold+1) % folds else 'train')
                parts[part].append(p)
    return dict(schema_version=1, seed=seed, fold=fold, folds=folds if fold is not None else None,
                physical_ids={k: sorted(v) for k,v in parts.items()})

def apply_splits(rows, splits, classes):
    pmap = {}
    for part in ['train','val','test']:
        for p in splits['physical_ids'][part]:
            if p in pmap: raise ValueError('Specimen appears in more than one split: '+p)
            pmap[p] = part
    if set(pmap) != {r['physical_id'] for r in rows}:
        raise ValueError('Split specimens do not exactly match eligible dataset. Explicitly regenerate splits for new specimens.')
    for r in rows: r['partition'] = pmap[r['physical_id']]
    for part in ['train','val','test']:
        found = {r['label'] for r in rows if r['partition']==part}
        if found != set(classes): raise ValueError(f'{part} lacks class(es) {set(classes)-found}; revise specimen splits.')

def validate_geometry(rows, colour=False):
    for r in rows:
        p = Path(r['folder'])/'geometry.npz'
        if not p.is_file(): raise FileNotFoundError(p)
        expected = r.get('geometry_sha256')
        if not expected: raise ValueError('Missing geometry provenance hash: '+r['scan_id'])
        if digest(p) != expected: raise ValueError('Geometry changed after preparation: '+str(p))
        with np.load(p, allow_pickle=False) as z:
            v, f, n = z['vertices'], z['faces'], z['normals']
            if v.ndim != 2 or v.shape[1]!=3 or n.shape!=v.shape or f.ndim!=2 or f.shape[1]!=3:
                raise ValueError('Invalid geometry dimensions: '+r['scan_id'])
            if len(f)<16 or not np.isfinite(v).all() or not np.isfinite(n).all():
                raise ValueError('Nonfinite or insufficient geometry: '+r['scan_id'])
            if not np.issubdtype(f.dtype, np.integer) or f.min()<0 or f.max()>=len(v):
                raise ValueError('Invalid triangle indices: '+r['scan_id'])
            if colour:
                c=z['vertex_rgb']
                if c.shape!=v.shape or not np.isfinite(c).all() or c.min()<0 or c.max()>1:
                    raise ValueError('RGB must be finite Nx3 in [0,1].')

def geometry(row):
    with np.load(Path(row['folder'])/'geometry.npz', allow_pickle=False) as z:
        v=z['vertices'].astype('float32'); f=z['faces'].astype('int64'); n=z['normals'].astype('float32')
        c=z['vertex_rgb'].astype('float32') if 'vertex_rgb' in z else np.zeros_like(v)
    norm=np.linalg.norm(n,axis=1,keepdims=True)
    n=n/np.maximum(norm,1e-8)
    area=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)/2
    if not np.isfinite(area).all() or area.sum()<=0: raise ValueError('No finite positive surface area.')
    return v,f,n,c,area

def inputs(row, model, n_elements, colour, scale_mm, seed, epoch=0, augment=False):
    v,f,n,c,area=geometry(row)
    rng=np.random.default_rng(seed_for(row['scan_id'],seed+epoch*100003))
    if model=='tsgcnet':
        valid=np.flatnonzero(area>1e-12)
        if len(valid)<16: raise ValueError('Fewer than 16 nondegenerate faces.')
        # Uniform sampling has equal inclusion probabilities, so area-weighted pooling is valid.
        ids=np.sort(rng.choice(valid,min(len(valid),n_elements),replace=False))
        p=v[f[ids]].copy(); normals=n[f[ids]].copy(); rgb=c[f[ids]].copy()
        local=dict(face_ids=ids, area_mm2=area[ids].copy(), positions_mm=p.mean(1).copy())
    else:
        ids=rng.choice(len(f),n_elements,p=area/area.sum())
        u=rng.random((n_elements,2)); s=np.sqrt(u[:,0]); bary=np.stack([1-s,s*(1-u[:,1]),s*u[:,1]],1)
        p=(v[f[ids]]*bary[:,:,None]).sum(1).astype('float32')
        normals=(n[f[ids]]*bary[:,:,None]).sum(1).astype('float32')
        normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-8)
        rgb=(c[f[ids]]*bary[:,:,None]).sum(1).astype('float32')
        local=dict(face_ids=ids,barycentric=bary.astype('float32'), positions_mm=p.copy())
    if augment:
        a=rng.uniform(-np.pi,np.pi); rot=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]],'float32')
        p=p@rot.T; normals=normals@rot.T
    p=p/scale_mm  # One constant across all specimens; no per-scan height standardisation.
    if model=='tsgcnet':
        streams=[p.reshape(-1,9),normals.reshape(-1,9)]
        if colour: streams.append(rgb.reshape(-1,9)-.5)
        batch=dict(xyz=torch.from_numpy(p.mean(1))[None],
                   streams=[torch.from_numpy(x.T.copy())[None] for x in streams],
                   area=torch.from_numpy(area[local['face_ids']].copy())[None])
    else:
        feats=[p,normals]
        if colour: feats.append(rgb-.5)
        batch=dict(xyz=torch.from_numpy(p)[None], features=torch.from_numpy(np.concatenate(feats,1).T.copy())[None])
    return batch,local

def to_device(batch,device):
    return {k:[x.to(device) for x in v] if isinstance(v,list) else v.to(device) for k,v in batch.items()}
