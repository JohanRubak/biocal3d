"""Shared, geometry-only ROI selection. Coordinates are never modified."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ALGORITHM_VERSION = 'surface_faces_v1'
DEFAULT_HEIGHT_HALF_WIDTH = 0.5

@dataclass
class Extraction:
    cx: float = 0.
    cy: float = 0.
    z: float = 0.
    radius: float = 5.
    tilt_x: float = 0.
    tilt_y: float = 0.
    below: float = DEFAULT_HEIGHT_HALF_WIDTH
    above: float = DEFAULT_HEIGHT_HALF_WIDTH
    max_angle: float = 75.
    connected: bool = True

    def validate(self):
        values = [v for k,v in asdict(self).items() if k != 'connected']
        if not np.isfinite(values).all() or self.radius <= 0 or self.below < 0 or self.above < 0:
            raise ValueError('Extraction values must be finite; radius positive and height limits nonnegative.')
        if not 0 < self.max_angle <= 90 or abs(self.tilt_x) >= 85 or abs(self.tilt_y) >= 85:
            raise ValueError('Invalid angle/plane tilt.')
        return self

    def frame(self):
        self.validate()
        tx,ty=np.deg2rad([self.tilt_x,self.tilt_y])
        n=np.array([np.tan(ty),-np.tan(tx),1.]); n/=np.linalg.norm(n)
        u=np.cross([0.,1.,0.],n); u/=np.linalg.norm(u)
        v=np.cross(n,u)
        return np.array([self.cx,self.cy,self.z]),u,v,n


def face_geometry(vertices, faces):
    t=np.asarray(vertices)[np.asarray(faces)]
    cross=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]); length=np.linalg.norm(cross,axis=1)
    normals=cross/np.maximum(length[:,None],1e-30)
    return t.mean(axis=1), normals, length/2


def edge_regions(faces, mask):
    """Connectivity through a shared edge, not an isolated touching vertex."""
    ids=np.flatnonzero(mask)
    if not len(ids):return ids,np.empty(0,int)
    f=np.asarray(faces)[ids]
    edges=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
    owner=np.tile(np.arange(len(ids)),3)
    order=np.lexsort((edges[:,1],edges[:,0])); edges=edges[order]; owner=owner[order]
    same=np.all(edges[1:]==edges[:-1],axis=1)
    a,b=owner[:-1][same],owner[1:][same]
    graph=coo_matrix((np.ones(2*len(a)),(np.r_[a,b],np.r_[b,a])),shape=(len(ids),len(ids)))
    _,labels=connected_components(graph,directed=False)
    return ids,labels


def select_surface(vertices,faces,settings):
    s=settings.validate(); c,u,v,n=s.frame()
    vertices=np.asarray(vertices,float); faces=np.asarray(faces,int)
    relative=vertices-c
    x,y,h=relative@u,relative@v,relative@n
    radial_inside=x*x+y*y <= s.radius*s.radius+1e-10
    inside=radial_inside&(h>=-s.below)&(h<=s.above)
    centers,normals,areas=face_geometry(vertices,faces)
    alignment=np.abs(normals@n) # tolerate inconsistent triangle winding
    broad=inside[faces].all(axis=1)&(areas>1e-14)
    oriented=broad&(alignment>=np.cos(np.deg2rad(s.max_angle))-1e-12)
    mask=oriented.copy(); ids,regions=edge_regions(faces,mask)
    components=int(regions.max()+1) if len(regions) else 0
    if s.connected and components:
        rel=centers-c
        central=(rel@u)**2+(rel@v)**2 <= (.55*s.radius)**2
        central_area=np.bincount(regions,weights=areas[ids]*central[ids],minlength=components)
        total=np.bincount(regions,weights=areas[ids],minlength=components)
        eligible=np.flatnonzero(central_area>0)
        if not len(eligible):
            mask[:]=False
        else:
            # Prefer a broad central sheet over dense tiny fragments.
            selected=eligible[np.argmax(total[eligible])]
            mask[ids[regions!=selected]]=False
    kept_area=float(areas[mask].sum()); candidate_area=float(areas[broad].sum())
    flags=[]
    if not mask.any():flags.append('empty_surface')
    if components>1:flags.append('multiple_surface_components')
    if np.sum(broad&~oriented):flags.append('steep_faces_removed')
    fraction=kept_area/candidate_area if candidate_area else 0.
    if fraction<.8:flags.append('substantial_geometry_removed')
    metrics=dict(algorithm=ALGORITHM_VERSION,radius_faces=int(radial_inside[faces].all(axis=1).sum()),candidate_faces=int(broad.sum()),oriented_faces=int(oriented.sum()),retained_faces=int(mask.sum()),
                 components=components,retained_area_mm2=kept_area,retained_fraction_of_candidate_area=fraction,
                 flags=flags,review_status='pending')
    return mask,metrics
