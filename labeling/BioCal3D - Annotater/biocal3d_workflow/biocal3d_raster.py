"""Triangle rasterizer adapted from biocal3d_prepare_dino.py."""
import numpy as np
def triangles(raw):
    f = np.asarray(raw, dtype=np.int64)
    if f.ndim == 2 and f.shape[1] == 3:
        return f
    if f.ndim == 2 and f.shape[1] == 4 and np.all(f[:, 0] == 3):
        return f[:, 1:]
    if f.ndim == 1 and f.size % 4 == 0 and np.all(f.reshape(-1, 4)[:, 0] == 3):
        return f.reshape(-1, 4)[:, 1:]
    raise ValueError('Missing or unsupported triangle connectivity; use roi.vtp.')

def sample_texture(tex, uv, flip_v=True):
    if np.any(uv < -0.0001) or np.any(uv > 1.0001):
        raise ValueError('UV outside [0,1]; repeat/wrap textures require an explicit mapping rule.')
    uv = np.clip(uv, 0, 1)
    x = uv[:, 0] * (tex.shape[1] - 1)
    y = (1 - uv[:, 1] if flip_v else uv[:, 1]) * (tex.shape[0] - 1)
    x0, y0 = (x.astype(int), y.astype(int))
    x1, y1 = (np.minimum(x0 + 1, tex.shape[1] - 1), np.minimum(y0 + 1, tex.shape[0] - 1))
    a, b = ((x - x0)[:, None], (y - y0)[:, None])
    return (1 - b) * ((1 - a) * tex[y0, x0] + a * tex[y0, x1]) + b * ((1 - a) * tex[y1, x0] + a * tex[y1, x1])

def rasterize(mesh, xyz, size, fov, flip_v=True):
    px = np.column_stack([(xyz[:, 0] / fov + 0.5) * size - 0.5, (0.5 - xyz[:, 1] / fov) * size - 0.5])
    tid = np.full((size, size), -1, np.int32)
    bary = np.zeros((size, size, 3), np.float32)
    zbuf = np.full((size, size), -np.inf)
    overlap = np.zeros((size, size), np.uint16)
    for i, face in enumerate(mesh['faces']):
        a, b, c = px[face]
        lo = np.maximum(np.ceil(np.min([a, b, c], axis=0)).astype(int), 0)
        hi = np.minimum(np.floor(np.max([a, b, c], axis=0)).astype(int), size - 1)
        if np.any(lo > hi):
            continue
        det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(det) < 1e-12:
            continue
        xx, yy = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        wa = ((b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])) / det
        wb = ((c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])) / det
        wc = 1 - wa - wb
        inside = (wa >= -1e-07) & (wb >= -1e-07) & (wc >= -1e-07)
        weights = np.stack([wa, wb, wc], axis=-1)
        depth = weights @ xyz[face, 2]
        overlap[yy[inside], xx[inside]] += 1
        take = inside & (depth > zbuf[yy, xx])
        rr, cc = (yy[take], xx[take])
        zbuf[rr, cc] = depth[take]
        tid[rr, cc] = i
        bary[rr, cc] = weights[take]
    valid = tid >= 0
    if valid.sum() == 0:
        raise ValueError('Empty projected map')
    faces = mesh['faces'][tid[valid]]
    w = bary[valid]
    rgb = np.full((size, size, 3), [0.485, 0.456, 0.406], np.float32)
    if mesh['texture'] is not None:
        uv = np.einsum('ni,nij->nj', w, mesh['uv'][faces])
        rgb[valid] = sample_texture(mesh['texture'], uv, flip_v)
    else:
        rgb[valid] = np.einsum('ni,nij->nj', w, mesh['rgb'][faces])
    height = np.where(valid, zbuf, np.nan).astype(np.float32)
    tr = xyz[mesh['faces']]
    cross = np.cross(tr[:, 1] - tr[:, 0], tr[:, 2] - tr[:, 0])
    ratio = np.linalg.norm(cross, axis=1) / np.maximum(np.abs(cross[:, 2]), 1e-12)
    area = np.zeros((size, size), np.float32)
    area[valid] = (fov / size) ** 2 * ratio[tid[valid]]
    return dict(rgb=np.clip(rgb, 0, 1), valid=valid, height_mm=height, triangle_id=tid, barycentric=bary, pixel_surface_area_mm2=area, overlap_fraction=float(np.mean(overlap[valid] > 1)))
