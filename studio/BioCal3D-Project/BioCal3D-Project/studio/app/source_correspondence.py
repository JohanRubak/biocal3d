"""Verify triangle geometry across loaders; retain provenance face numbering."""
from itertools import permutations
import numpy as np
from scipy.spatial import cKDTree

PERMUTATIONS=tuple(permutations(range(3)))

def corner_order(source,target,tolerance):
    source=np.asarray(source,float);target=np.asarray(target,float)
    order=np.full((len(target),3),-1,np.int64);found=np.zeros(len(target),bool)
    for perm in PERMUTATIONS:
        match=np.all(np.linalg.norm(source[:,perm,:]-target,axis=2)<=tolerance,axis=1)&~found
        order[match]=perm;found|=match
    return order,found


def triangle_correspondence(source,target,tolerance=2e-5):
    """Target face -> source face/corners. Reject missing or ambiguous matches.

    Index equality is not required. Every triangle's three coordinates must
    agree within tolerance, with a one-to-one face mapping. This function is
    used for appearance transfer, never to infer plaque labels spatially.
    """
    source=np.asarray(source,float);target=np.asarray(target,float)
    if source.shape!=target.shape or source.ndim!=3 or source.shape[1:]!=(3,3):
        raise ValueError('Original loaders produced different triangle counts; source correspondence cannot be verified.')
    if not np.isfinite(source).all() or not np.isfinite(target).all():raise ValueError('Invalid source coordinates.')
    corners,matched=corner_order(source,target,tolerance)
    rows=np.arange(len(target),dtype=np.int64)
    if matched.all():return rows,corners
    used=set();tree=cKDTree(source.mean(axis=1))
    # Once row order differs, verify every face independently; coincident
    # duplicates cannot inherit appearance through an arbitrary pairing.
    for target_id in range(len(target)):
        candidates=[i for i in tree.query_ball_point(target[target_id].mean(axis=0),tolerance*1.01) if i not in used]
        if not candidates:raise ValueError('Original loader triangles differ geometrically; safe source mapping is unavailable.')
        orders,valid=corner_order(source[candidates],np.repeat(target[target_id][None],len(candidates),axis=0),tolerance)
        hits=np.flatnonzero(valid)
        if len(hits)!=1:raise ValueError('Original triangle correspondence is missing or ambiguous; no labels or textures were reassigned.')
        hit=int(hits[0]);rows[target_id]=candidates[hit];corners[target_id]=orders[hit];used.add(candidates[hit])
    return rows,corners
