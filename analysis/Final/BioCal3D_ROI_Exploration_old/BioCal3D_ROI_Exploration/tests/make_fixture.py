"""Small synthetic prepared dataset for tests. Not biological data or plaque ground truth."""
from pathlib import Path
import json
import numpy as np
import matplotlib.tri as mtri


def make_dataset(root, specimens=2, scanners=("iTERO", "TRIOS3", "TRIOS5", "LABscanner")):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    for group in [1, 3]:
        stages = [0, 1] if group == 1 else [0, 1, 2]
        for number in range(1, specimens+1):
            for si, scanner in enumerate(scanners):
                for t in stages:
                    sample = f"BCG{group}T{t}-{number}"
                    folder = root/"samples"/scanner/f"G{group}"/sample; folder.mkdir(parents=True)
                    n = 9+si*2
                    x = np.linspace(-.8, .8, n); xx, yy = np.meshgrid(x, x)
                    h = .012*np.sin(6*xx)*np.cos(5*yy)*(1-.15*t)*(1+.1*si)+.004*xx-.002*yy
                    v = np.column_stack([xx.ravel(), yy.ravel(), h.ravel()])
                    a = np.arange(n*n).reshape(n, n)[:-1, :-1].ravel()
                    f = np.concatenate([np.c_[a, a+1, a+n], np.c_[a+1, a+n+1, a+n]]).astype(np.int64)
                    np.savez_compressed(folder/"geometry.npz", vertices=v, faces=f)
                    tri = mtri.Triangulation(v[:, 0], v[:, 1], f)
                    size, fov = 64, 2.
                    coords = (np.arange(size)+.5)*fov/size-fov/2; X, Y = np.meshgrid(coords, -coords)
                    ids = tri.get_trifinder()(X, Y); valid = ids >= 0
                    height = np.asarray(mtri.LinearTriInterpolator(tri, v[:, 2])(X, Y).filled(np.nan))
                    rgb = np.ones((size, size, 3), np.float32)*.5
                    rgb[..., 0] = .65+.015*si+.025*t+.01*number+.02*X
                    rgb[..., 1] = .30+.08*t-.01*si+.02*Y
                    rgb[..., 2] = .40+.035*t+.005*si
                    tr = v[f]; cross = np.cross(tr[:, 1]-tr[:, 0], tr[:, 2]-tr[:, 0])
                    ratio = np.linalg.norm(cross, axis=1)/np.abs(cross[:, 2])
                    area = np.zeros_like(height); area[valid] = (fov/size)**2*ratio[ids[valid]]
                    np.savez_compressed(folder/"maps.npz", vertices=v, faces=f, rgb=rgb, valid=valid, height_mm=height,
                                        pixel_surface_area_mm2=area, fov_mm=fov)
                    stage = ("baseline" if t == 0 else "clean") if group == 1 else ["baseline", "partial", "clean"][t]
                    meta = dict(scanner=scanner, stage=stage, geometry_review_status="approved", color_source="vertex_rgb",
                                split="train" if number == 1 else "test", fov_mm=fov)
                    (folder/"metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (root/"run_config.json").write_text(json.dumps({"fov_mm": 2., "synthetic_fixture": True}))
    return root


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("output", type=Path)
    args = p.parse_args(); print(make_dataset(args.output))
