"""Convert existing prepared ROIs into a separate frozen PLY annotation dataset."""
import argparse
from pathlib import Path
from mesh_io import discover, load_mesh, prepare_roi_ply

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,help='Existing prepared ROI dataset, not raw holder scans')
    parser.add_argument('--output',required=True,help='New empty output directory, separate from input')
    parser.add_argument('--contains',default='',help='Optional relative-path substring filter')
    args=parser.parse_args()
    root=Path(args.root).resolve(); output=Path(args.output).resolve()
    if output==root or output.is_relative_to(root) or root.is_relative_to(output):
        parser.error('Choose a separate output directory, outside the input tree.')
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory must be empty; preparing again must not replace frozen annotated meshes.')
    sources=discover(root,args.contains)
    if not sources:parser.error('No supported prepared meshes found.')
    failures=0
    for i,source in enumerate(sources,1):
        # Include source stem so directories holding several scans cannot collide.
        relative=source.relative_to(root)
        target=output/relative.parent/(source.stem+'.ply')
        try:
            mesh=load_mesh(source)
            if mesh.warning:raise ValueError(mesh.warning)
            if mesh.imported_labels is not None:
                raise ValueError('Input already contains labels. Use the frozen original dataset; do not recrop labeled inputs.')
            prepare_roi_ply(mesh,target)
            print(f'[{i}/{len(sources)}] OK {relative}: {mesh.active.sum():,} ROI triangles -> {target.relative_to(output)}')
        except Exception as exc:
            failures+=1;print(f'[{i}/{len(sources)}] FAILED {relative}: {exc}')
    print(f'Finished: {len(sources)-failures} converted, {failures} failed. Input files unchanged.')
    raise SystemExit(1 if failures else 0)

if __name__=='__main__':main()
