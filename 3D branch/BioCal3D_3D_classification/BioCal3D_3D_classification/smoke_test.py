"""Generate a small synthetic snapshot and verify the complete CPU workflow.

This tests software plumbing only; synthetic scores have no scientific meaning.
Run: python smoke_test.py --output path/to/smoke_results
"""
from pathlib import Path
import argparse, json, subprocess, sys
import numpy as np
from PIL import Image
from data3d import digest,save_json


def make_fixture(root):
    root.mkdir(parents=True,exist_ok=True); rows=[]
    n=9; axis=np.linspace(-5,5,n); xx,yy=np.meshgrid(axis,axis)
    faces=[]
    for i in range(n-1):
        for j in range(n-1):
            a=i*n+j; faces += [[a,a+1,a+n+1],[a,a+n+1,a+n]]
    faces=np.array(faces,dtype='int64')
    for group in [1,3]:
        for specimen in range(1,7):
            for stage_idx,stage in enumerate(['baseline','partial','clean']):
                for scanner in ['TRIOS3','PRIMESCAN']:
                    sid=f'{scanner}__BCG{group}T{stage_idx}-{specimen}'
                    folder=root/'samples'/sid; folder.mkdir(parents=True,exist_ok=True)
                    z=(.1+.12*(2-stage_idx))*np.exp(-(xx**2+yy**2)/12)+.001*specimen
                    v=np.stack([xx,yy,z],-1).reshape(-1,3).astype('float32')
                    dx=-xx/6*(z-.001*specimen); dy=-yy/6*(z-.001*specimen)
                    normals=np.stack([-dx,-dy,np.ones_like(dx)],-1).reshape(-1,3).astype('float32'); normals/=np.linalg.norm(normals,axis=1,keepdims=True)
                    colour=np.broadcast_to(np.array([[.35+.17*stage_idx,.4,.6-.13*stage_idx]],'float32'),v.shape).copy()
                    np.savez_compressed(folder/'geometry.npz',vertices=v,faces=faces,normals=normals,vertex_rgb=colour,
                        vertices_raw=v,raw_vertex_id=np.arange(len(v)),raw_triangle_id=np.arange(len(faces)),
                        original_to_canonical=np.eye(4),canonical_to_original=np.eye(4),roi_mask=np.ones(len(v),bool))
                    tid=np.arange(0,len(faces),2).reshape(n-1,n-1).astype('int32'); valid=np.ones_like(tid,bool)
                    bary=np.broadcast_to(np.array([.5,0,.5]),(*tid.shape,3)).astype('float32')
                    np.savez_compressed(folder/'maps.npz',vertices=v,faces=faces,triangle_id=tid,valid=valid,barycentric=bary)
                    Image.fromarray(np.tile(np.uint8(np.rint(colour[0]*255)),(n-1,n-1,1))).save(folder/'rgb.png')
                    rows.append(dict(scan_id=sid,physical_id=f'G{group}_S{specimen}',scanner=scanner,group=group,
                        stage=stage,status='OK',time_idx=stage_idx,timepoint='T'+str(stage_idx),sample_number=specimen,
                        growth_condition='fresh_12h' if group==1 else 'mature_24h',sample_dir=folder.relative_to(root).as_posix(),
                        geometry_sha256=digest(folder/'geometry.npz'),maps_sha256=digest(folder/'maps.npz'),rgb_sha256=digest(folder/'rgb.png')))
    save_json(root/'dataset_index.json',dict(schema_version=1,config=dict(fov_mm=13.4,coordinate_units='mm',synthetic=True),scans=rows))
    return root/'dataset_index.json'


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    root=args.output.resolve()
    if root.exists(): raise ValueError('Use a new output path for the smoke test.')
    index=make_fixture(root/'snapshot'); result=root/'results'
    script=Path(__file__).with_name('biocal3d_train_3d.py')
    command=[sys.executable,str(script),'run','--index',str(index),'--output',str(result),'--device','cpu',
        '--epochs','2','--patience','2','--batch-size','4','--faces','32','--points','32','--threads','1',
        '--bootstrap','30','--explain-count','2']
    subprocess.run(command,check=True)
    checks={}
    assignments=json.loads((result/'splits.json').read_text())['physical_ids']
    assert not (set(assignments['train']) & set(assignments['test']))
    assert not (set(assignments['train']) & set(assignments['val']))
    assert not (set(assignments['test']) & set(assignments['val']))
    import torch
    from models3d import build
    from data3d import load_index,inputs,to_device
    rows,classes,_=load_index(index)
    torch.set_num_threads(1)
    for folder in sorted(result.glob('*_seed42')):
        ckpt=torch.load(folder/'best.pt',map_location='cpu',weights_only=False); cfg=ckpt['config']
        net=build(cfg['model'],len(classes),cfg['colour']); net.load_state_dict(ckpt['state_dict']); net.eval()
        batch,_=inputs(rows[0],cfg['model'],32,cfg['colour'],6.7,cfg['sampling_seed'])
        net.zero_grad(); logits,features,_=net(batch,explain=True); logits[0,0].backward()
        assert features.grad is not None and torch.isfinite(features.grad).all() and features.grad.abs().sum()>0
        signed=(features.grad.mean(-1,keepdim=True)*features).sum(1).detach().numpy()
        # The explanation must depend on the trained classifier head.
        with torch.no_grad():
            net.classifier[-1].weight.neg_(); net.classifier[-1].bias.neg_()
        net.zero_grad(); changed,f2,_=net(batch,explain=True); changed[0,0].backward()
        signed2=(f2.grad.mean(-1,keepdim=True)*f2).sum(1).detach().numpy()
        assert np.allclose(signed,-signed2,atol=1e-6,rtol=1e-3) and np.max(np.abs(signed-signed2))>0
        for file in folder.glob('explanations/*/attribution.npz'):
            with np.load(file) as z:
                assert np.isfinite(z['cam']).all() and z['cam'].min()>=0 and z['cam'].max()<=1.00001
                if cfg['model']=='tsgcnet': assert np.isfinite(z['face_cam']).sum()==32
            overlay=file.with_name('overlay.png'); assert overlay.exists() and overlay.stat().st_size>1000
            assert file.with_name('surface.html').stat().st_size>10000
            import xml.etree.ElementTree as ET
            ET.parse(file.with_name('attribution.vtp'))
        assert len(json.loads((folder/'history.json').read_text()))==2
        assert (folder/'encoder.pt').is_file()
        checks[folder.name]=dict(nonzero_local_gradient=True,head_dependent_cam=True,trained_epochs=2,outputs_verified=True)
    assert len(checks)==4
    assert (result/'comparison.html').is_file() and (result/'paired_differences.csv').is_file()
    # Provenance guard must reject altered geometry.
    from data3d import validate_geometry
    first=Path(rows[0]['folder'])/'geometry.npz'; original=first.read_bytes(); first.write_bytes(original+b'changed')
    try:
        validate_geometry(rows[:1]); raise AssertionError('Expected geometry mismatch rejection')
    except ValueError: pass
    finally: first.write_bytes(original)
    save_json(root/'VALIDATION.json',dict(synthetic_only=True,checks=checks,
        specimen_split_disjoint=True,geometry_hash_mismatch_rejected=True,
        torch_version=torch.__version__,command=command))
    print('PASS. Synthetic workflow verified:',root/'VALIDATION.json')

if __name__=='__main__': main()
