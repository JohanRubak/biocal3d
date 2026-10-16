"""Copy pilot ROI inputs into portable calibration/agreement folders."""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path
import numpy as np
import vtk
from annotator.mesh_io import load_mesh, discover
from prepare_pilot_study import prepare, load_inventory, digest, write_csv, PROFILES


def patch_ply_texture(source, target, texture_name):
    # Preserve every byte of the vertex/face payload, including UV seams.
    with source.open('rb') as src, target.open('wb') as dst:
        lines=[];size=0
        while True:
            line=src.readline();size+=len(line)
            if not line or size>1024*1024:raise ValueError(f'Invalid PLY header: {source}')
            if line.strip()==b'end_header':break
            if not line.lower().startswith(b'comment texturefile '):lines.append(line)
        newline=b'\r\n' if line.endswith(b'\r\n') else b'\n'
        # PlyData exposes only file-level comments through d.comments. Put
        # TextureFile before the first element, rather than under face properties.
        inserted=False
        for header_line in lines:
            dst.write(header_line)
            if header_line.lower().startswith(b'format '):
                dst.write(b'comment TextureFile '+texture_name.encode('ascii')+newline);inserted=True
        if not inserted:raise ValueError(f'PLY format missing: {source}')
        dst.write(line);shutil.copyfileobj(src,dst)


def copy_scan(root, destination, row):
    source=(root/row['annotation_mesh']).resolve()
    relative=source.relative_to(root).with_name('geometry.npz' if source.suffix.lower()=='.npz' else 'roi'+source.suffix.lower())
    target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise ValueError(f'Multiple scans use the same sample folder: {relative.parent}')
    mesh=load_mesh(source)
    if mesh.imported_labels is not None and np.any(mesh.imported_labels[mesh.active]!=-2):
        raise ValueError(f'Input already contains ROI labels; use the original unlabeled ROI: {source}')
    if mesh.warning:raise ValueError(f'Cannot package unverified color for {source}: {mesh.warning}')
    texture_name=None
    if mesh.texture is not None:
        texture_name=target.stem+'_texture'+mesh.texture.suffix.lower()
        shutil.copy2(mesh.texture,target.with_name(texture_name))
    if texture_name and source.suffix.lower()=='.ply':
        patch_ply_texture(source,target,texture_name)
    elif texture_name and source.suffix.lower()=='.vtp':
        reader=vtk.vtkXMLPolyDataReader();reader.SetFileName(str(source));reader.Update()
        poly=reader.GetOutput();poly.GetFieldData().RemoveArray('TextureFile')
        field=vtk.vtkStringArray();field.SetName('TextureFile');field.InsertNextValue(texture_name)
        poly.GetFieldData().AddArray(field)
        writer=vtk.vtkXMLPolyDataWriter();writer.SetFileName(str(target));writer.SetInputData(poly)
        if writer.Write()!=1:raise OSError(f'Cannot write portable ROI: {target}')
    elif texture_name and source.suffix.lower()=='.npz':
        with np.load(source,allow_pickle=False) as data:arrays={name:data[name] for name in data.files}
        arrays['texture_path']=np.array(texture_name)
        if 'texture_file' in arrays:arrays['texture_file']=np.array(texture_name)
        np.savez_compressed(target,**arrays)
    else:shutil.copy2(source,target)
    # Canonical transform is used for the top/bottom indicator. Only this
    # supplemental geometry file is copied; never a second PLY/VTP input.
    geometry=source.with_name('geometry.npz')
    if geometry.is_file() and geometry!=source:
        with np.load(geometry,allow_pickle=False) as data:has_labels=bool({'face_labels','face_label'}&set(data.files))
        if has_labels:
            supplemental=load_mesh(geometry)
            if np.any(supplemental.imported_labels[supplemental.active]!=-2):raise ValueError(f'Supplemental geometry contains labels: {geometry}')
        shutil.copy2(geometry,target.with_name('geometry.npz'))
    for name in ('qc.png','context_qc.png'):
        extra=source.with_name(name)
        if extra.is_file():shutil.copy2(extra,target.with_name(name))
    copied=load_mesh(target)
    if copied.fingerprint!=mesh.fingerprint:raise ValueError(f'Geometry/face order changed while copying {source}')
    if copied.warning:raise ValueError(f'Portable color not verified for {target}: {copied.warning}')
    if texture_name and (copied.texture is None or not copied.texture.resolve().is_relative_to(destination.resolve())):
        raise ValueError(f'Copied mesh still depends on an external texture: {target}')
    updated={**row,'annotation_mesh':relative.as_posix(),'sample_dir':relative.parent.as_posix(),
             'mesh_fingerprint':mesh.fingerprint}
    if target.suffix.lower()=='.ply':updated['ply_sha256']=digest(target)
    updated['texture_path']=(relative.parent/texture_name).as_posix() if texture_name else ''
    metadata_source=source.with_name('metadata.json')
    metadata=json.loads(metadata_source.read_text(encoding='utf-8')) if metadata_source.is_file() else {}
    if not isinstance(metadata,dict):raise ValueError(f'Metadata must be a JSON object: {metadata_source}')
    metadata.update(updated)
    metadata.update(portable_dataset=True,texture_path=texture_name or '',
                    source_annotation_mesh_provenance=str(source))
    if 'ply_texture' in metadata:metadata['ply_texture']=texture_name or ''
    metadata.setdefault('sample_id',relative.parent.name)
    target.with_name('metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return updated


def export_shared(root, output=None, plan=None, calibration=None, agreement=None,
                  agreement_g5=None, seed=42, min_scanners=4,
                  calibration_ids=None, agreement_ids=None):
    root=Path(root).resolve()
    output=Path(output or root.parent/'G5_labeling_study').resolve()
    if output.exists():raise ValueError('Output already exists. Choose a new --output; existing labels/data will not be overwritten.')
    if root.is_relative_to(output) or output.is_relative_to(root/'samples') or output.is_relative_to(root/'annotations'):
        raise ValueError('Put the study folder outside source samples and working annotations.')
    overrides=any(v is not None for v in (calibration,agreement,agreement_g5,calibration_ids,agreement_ids))
    if plan is None and not overrides and (root/'labeling_g5_pilot'/'plan.json').is_file():plan=root/'labeling_g5_pilot'
    reused=False
    if plan:
        if overrides:raise ValueError('Use --plan to preserve selection OR specify a new selection, not both.')
        config=json.loads((Path(plan)/'plan.json').read_text(encoding='utf-8'))
        if config.get('inventory_sha256')!=digest(root/'annotation_inventory.csv'):
            raise ValueError('Existing plan inventory differs from this snapshot; inspect the change before exporting.')
        calibration_ids=config['calibration_ids'];agreement_ids=config['agreement_ids']
        calibration=len(calibration_ids);agreement=len(agreement_ids)
        grouped={r['physical_id']:r['group'] for r in load_inventory(root)}
        agreement_g5=sum(grouped[pid]=='5' for pid in agreement_ids)
        seed=config.get('seed',seed);reused=True
    with tempfile.TemporaryDirectory() as scratch,contextlib.redirect_stdout(io.StringIO()):
        assigned,config=prepare(root,Path(scratch)/'selection',2 if calibration is None else calibration,
                                6 if agreement is None else agreement,agreement_g5,
                                min_scanners=min_scanners,seed=seed,calibration_ids=calibration_ids,
                                agreement_ids=agreement_ids,write_launchers=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.g5_export_',dir=output.parent) as temporary:
        staging=Path(temporary)/'study';staging.mkdir()
        for phase in ('calibration','agreement'):
            destination=staging/phase;destination.mkdir()
            inputs=[r for r in assigned if r['study_phase']==phase];phase_rows=[]
            print(f'Copying {phase}: {len(inputs)} scans',flush=True)
            for index,row in enumerate(inputs,1):
                phase_rows.append(copy_scan(root,destination,row))
                if index%10==0 or index==len(inputs):print(f'  {index}/{len(inputs)} copied and verified',flush=True)
            write_csv(destination/'annotation_inventory.csv',phase_rows)
            expected={(destination/r['annotation_mesh']).resolve() for r in phase_rows}
            if set(discover(destination))!=expected:
                raise ValueError(f'Opening the {phase} folder would select different/duplicate ROI inputs.')
            for name in PROFILES:
                folder=destination/'annotations'/name;folder.mkdir(parents=True)
                (folder/'profile.txt').write_text(f'Annotation files for {name}. Select your own name in the annotator.\n',encoding='utf-8')
            phase_config={**config,'root':'.','portable_dataset':True,'dataset_phase':phase,
                          'agreement_manifest':'annotation_inventory.csv',
                          'inventory_sha256':digest(destination/'annotation_inventory.csv'),
                          'source_root_provenance':str(root)}
            (destination/'plan.json').write_text(json.dumps(phase_config,indent=2),encoding='utf-8')
            (destination/'OPEN_THIS_FOLDER.txt').write_text(
                'In your normal annotator, click Open folder and select THIS folder.\n'
                'Select your own annotator name: bruna or katrin. Johan is experimental.\n'
                'Keep the samples, textures and annotations together when copying this folder.\n'
                'Labels are saved in annotations/<your_name>/ with matching relative sample paths.\n',encoding='utf-8')
        # These are source inventory records of scans not copied yet.
        write_csv(staging/'reserved_for_later.csv',[r for r in assigned if r['study_phase']=='production'])
        write_csv(staging/'selection.csv',[r for r in assigned if r['study_phase']!='production'])
        timing=[{'physical_id':pid,'expected_growth_hours':24,'actual_growth_hours':'','timing_status':'needs_recording','notes':''}
                for pid in sorted(set(config['calibration_ids']+config['agreement_ids']))]
        write_csv(staging/'specimen_timing_to_complete.csv',timing)
        (staging/'study_plan.json').write_text(json.dumps({**config,'root':'.','portable_folders':True,
                 'reused_existing_selection':reused,'source_root_provenance':str(root)},indent=2),encoding='utf-8')
        (staging/'README.txt').write_text(
            'Copy this whole study folder to the shared drive.\n'
            'Open calibration OR agreement in the annotator; select your own name.\n'
            'Open the phase folder itself, not samples or the parent study folder.\n'
            'Discuss calibration jointly, then complete agreement independently.\n'
            'Source annotations were not copied. Reserved scans are listed, not duplicated.\n'
            'After both raters finish, run analyse_labeling_agreement.py --root "PATH_TO_AGREEMENT_FOLDER"\n'
            'using the existing biocal3d environment. No original snapshot path is needed for analysis.\n',encoding='utf-8')
        staging.rename(output)
    print('Portable study:',output)
    print('Reused existing selection:',reused)
    print('Calibration:',config['counts']['calibration_bruna'])
    print('Agreement per dentist:',config['counts']['agreement_bruna'])
    print('Copy to your shared drive; open calibration or agreement and choose your own name.')
    return config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--plan',type=Path)
    p.add_argument('--calibration',type=int);p.add_argument('--agreement',type=int);p.add_argument('--agreement-g5',type=int)
    p.add_argument('--calibration-ids',nargs='+');p.add_argument('--agreement-ids',nargs='+')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--min-scanners',type=int,default=4)
    p.add_argument('--app-launcher',help=argparse.SUPPRESS);p.add_argument('--controls',type=int,default=0,help=argparse.SUPPRESS)
    a=p.parse_args()
    try:
        if a.controls:raise ValueError('All stages are included; remove --controls.')
        export_shared(a.root,a.output,a.plan,a.calibration,a.agreement,a.agreement_g5,a.seed,a.min_scanners,a.calibration_ids,a.agreement_ids)
        return 0
    except Exception as exc:print('ERROR:',exc,file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
