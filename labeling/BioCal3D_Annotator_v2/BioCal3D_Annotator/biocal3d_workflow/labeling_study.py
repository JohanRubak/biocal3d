"""Specimen-grouped calibration, independent agreement, consensus and production queues."""
from __future__ import annotations
import argparse,csv,json,random,subprocess,sys,hashlib,shutil,tempfile,os
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from annotator.mesh_io import AnnotationStore,load_mesh,manifest_queue,export_ply
from annotator.roi_review_store import read_reviewed,read_state,review_dir,current_version,digest


def create_study_folders(root,output=None,group=5,calibration=2,agreement=6,seed=42):
    """Copy frozen approved geometry, never plaque ratings, into portable subsets."""
    root=Path(root).resolve();output=Path(output or root.parent/f'G{group}_labeling_study').resolve()
    if root==output or root in output.parents or output in root.parents:
        raise ValueError('Study output must be separate from the source snapshot.')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Study output must be new/empty. Existing study labels will never be overwritten; use --output with a new name.')
    rebuilt=root/'geometry_rebuild.json'
    if (root/'INCOMPLETE_REBUILD.json').exists() or not rebuilt.is_file() or json.loads(rebuilt.read_text())['status']!='complete':
        raise ValueError('Use the successfully REBUILT approved snapshot, not the original preprocessing folder or an incomplete rebuild.')
    rows=[r for r in inventory(root) if str(r.get('group','')).upper().removeprefix('G')==str(group)]
    assigned=grouped_plan(rows,calibration,agreement,seed)
    selected=[r for r in assigned if r['study_phase'] in ('calibration','agreement')]
    store=AnnotationStore(root,root/'annotations','study_preparation');checks=[];seen_dirs=set()
    for row in selected:
        folder=(root/row['sample_dir']).resolve()
        if not folder.is_relative_to(root) or not folder.is_dir() or folder==root or folder in seen_dirs:
            raise ValueError('Invalid/duplicate sample directory: '+str(folder))
        seen_dirs.add(folder)
        canonical=root/row['annotation_mesh'];mesh=read_reviewed(store,canonical);state=read_state(store,mesh.path)
        if not state or state['status']!='approved':
            raise ValueError(f'{row.get("scan_id",canonical)} is not ROI-approved. Finish review and rebuild first.')
        baseline=load_mesh(canonical)
        if mesh.fingerprint!=baseline.fingerprint:
            raise ValueError(f'{row.get("scan_id",canonical)} changed after rebuilding. Rebuild again before creating study folders.')
        if baseline.imported_labels is not None:
            raise ValueError('Study input contains embedded plaque labels: '+str(canonical))
        if not canonical.resolve().is_relative_to(folder):raise ValueError('Annotation mesh is outside its sample directory.')
        if mesh.display.GetPointData().GetTCoords() is not None and not baseline.texture:
            raise ValueError('Missing portable ROI texture: '+str(canonical))
        checks.append((row,mesh.path,state['version'],digest(canonical)))
    # Do not silently omit other scanner/timepoint scans of a selected specimen.
    with (root/'scan_manifest.csv').open(encoding='utf-8-sig',newline='') as stream:all_scans=list(csv.DictReader(stream))
    pids={r['physical_id'] for r in selected};included={r['scan_id'] for r in selected}
    excluded=[r.get('scan_id','') for r in all_scans if r.get('physical_id') in pids and r.get('scan_id') not in included]
    if excluded:raise ValueError('Selected specimens have scans excluded from annotation_inventory.csv. Resolve their eligibility first: '+', '.join(excluded))
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.'+output.name+'_',dir=output.parent))
    counts={};audit=[]
    def ignore(folder,names):
        return [n for n in names if n in ('annotations','labels','annotation_output','feature_cache','surface_cache','__pycache__')
                or n.endswith(('.labels.npz','.labeled.ply','.labeled.vtp','.label_colors.ply','.lock','.tmp'))]
    try:
        for phase in ('calibration','agreement'):
            dest=staging/phase;dest.mkdir();subset=[r for r in selected if r['study_phase']==phase]
            counts[phase]={'specimens':len({r['physical_id'] for r in subset}),'scans':len(subset)}
            for row,source,version,checksum in checks:
                if row['study_phase']!=phase:continue
                shutil.copytree(root/row['sample_dir'],dest/row['sample_dir'],ignore=ignore)
                rel_review=review_dir(store,source).relative_to(store.output)
                shutil.copytree(review_dir(store,source),dest/'annotations'/rel_review,ignore=ignore)
                copied_store=AnnotationStore(dest,dest/'annotations','study_preparation')
                copied=read_reviewed(copied_store,dest/row['annotation_mesh'])
                original=read_reviewed(store,source)
                if copied.fingerprint!=original.fingerprint or copied.review_version!=version or copied.review_status!='approved':
                    raise ValueError('Copied approved surface differs: '+row['scan_id'])
                if digest(dest/row['annotation_mesh'])!=checksum:raise ValueError('ROI mesh changed during copy.')
                if original.texture and (not copied.texture or not copied.texture.is_relative_to(dest)):
                    raise ValueError('Copied ROI texture is not local: '+row['scan_id'])
                if original.texture and digest(original.texture)!=digest(copied.texture):
                    raise ValueError('Copied scanner texture differs: '+row['scan_id'])
                audit.append(dict(scan_id=row['scan_id'],physical_id=row['physical_id'],study_phase=phase,
                                  annotation_mesh=row['annotation_mesh'],roi_sha256=checksum,geometry_review_version=version))
            for rater in ('bruna','katrin'):(dest/'annotations'/rater).mkdir(parents=True)
            write_csv(dest/'annotation_inventory.csv',subset);write_csv(dest/'scan_manifest.csv',subset)
            for name in ('class_map.json','annotation_classes.json','classes.txt'):
                if (root/name).is_file():shutil.copy2(root/name,dest/name)
            config=dict(source_snapshot=str(root),group=group,phase=phase,seed=seed,geometry_frozen=True,plaque_labels_copied=False)
            (dest/'dataset_index.json').write_text(json.dumps(dict(schema_version=3,config=config,scans=subset),indent=2),encoding='utf-8')
            (dest/'study_subset.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
            # Existing agreement analysis can consume the portable agreement folder.
            plan=dest/'labeling_plan';plan.mkdir()
            write_csv(plan/'assignments.csv',subset)
            write_csv(plan/'agreement_dentist_1.csv',subset if phase=='agreement' else [])
            (plan/'plan.json').write_text(json.dumps(dict(root=str(output/phase),inventory_sha256=digest(dest/'annotation_inventory.csv'),
                 raters=['bruna','katrin'],counts=counts[phase],source_snapshot=str(root)),indent=2),encoding='utf-8')
            (dest/'README.txt').write_text(
                f'BioCal3D G{group} {phase}: {counts[phase]["specimens"]} specimens / {len(subset)} scans.\n'
                'Open THIS folder in the normal annotator, then select your own annotator name: bruna or katrin.\n'
                'ROI approvals are already present. Leave the approved geometry unchanged during the study.\n'
                'Plaque labels start empty and autosave separately under annotations/NAME.\n'
                'Use Approve Label & Next when a scan is complete.\n'
                'Discuss calibration before rating agreement independently; do not inspect the other dentist\'s agreement ratings.\n'
                'Move/copy the entire subset folder, including samples AND annotations.\n'
                'Original full scanner models are not bundled; ROI labeling needs only these portable inputs.\n',encoding='utf-8')
        for row,source,version,checksum in checks:
            if current_version(store,source)!=version or digest(root/row['annotation_mesh'])!=checksum:
                raise RuntimeError('Source ROI changed while preparing the study. Retry after closing the annotator.')
        write_csv(staging/'assignments.csv',selected)
        files={str(p.relative_to(staging)).replace('\\','/'):digest(p) for p in staging.rglob('*') if p.is_file()}
        (staging/'study_plan.json').write_text(json.dumps(dict(status='complete',source_snapshot=str(root),
            source_rebuild_sha256=digest(rebuilt),group=group,seed=seed,counts=counts,scans=audit,input_sha256=files,
            plaque_labels_copied=False,raters=['bruna','katrin']),indent=2),encoding='utf-8')
        if output.exists():output.rmdir()  # Empty output only; never replace an existing study.
        os.rename(staging,output)
    finally:
        if staging.exists():shutil.rmtree(staging)
    print('Study folders ready:',output);print(json.dumps(counts,indent=2));return counts


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row)) or ['annotation_mesh']
    with Path(path).open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,keys);writer.writeheader();writer.writerows(rows)


def resolve_root(value):
    if value:return Path(value).resolve()
    saved=Path(__file__).parent/'last_prepared.json'
    if saved.exists():return Path(json.loads(saved.read_text())['output']).resolve()
    from biocal3d_prepare_all import DEFAULT_ROOT
    if DEFAULT_ROOT is None:raise ValueError('Pass --root with your prepared snapshot.')
    choices=sorted(p for p in DEFAULT_ROOT.glob('_development_*') if (p/'annotation_inventory.csv').is_file())
    if not choices:raise ValueError('Pass --root with your existing prepared snapshot.')
    return choices[-1].resolve()


def inventory(root):
    path=root/'annotation_inventory.csv'
    with path.open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    if not rows:raise ValueError('The approved annotation inventory is empty.')
    # Validate hashes and paths once before assigning any case.
    approved=manifest_queue(root,path);by_path={str(p.relative_to(root)):p for p in approved}
    seen=set()
    for row in rows:
        if not row.get('annotation_mesh'):
            folder=root/row['sample_dir'];row['annotation_mesh']=next(str((folder/n).relative_to(root)) for n in ('roi.ply','roi.vtp') if (folder/n).is_file())
        rel=str(Path(row['annotation_mesh']))
        if rel not in by_path:raise ValueError(f'Invalid annotation mesh {rel}')
        if row['annotation_mesh'] in seen:raise ValueError('Duplicate mesh in inventory.')
        seen.add(row['annotation_mesh'])
        if not row.get('physical_id'):raise ValueError('Inventory must include physical_id to keep specimens together.')
    return rows


def grouped_plan(rows,calibration=4,agreement=8,seed=42,partial_fraction=.75):
    grouped=defaultdict(list)
    for row in rows:grouped[row['physical_id']].append(dict(row))
    if calibration<1 or agreement<1 or calibration+agreement>len(grouped):
        raise ValueError(f'{len(grouped)} physical specimens available; choose positive calibration/agreement counts whose sum is no greater than this.')
    rng=random.Random(seed);available=set(grouped)
    def stratum(pid):
        r=grouped[pid][0];return (r.get('group',''),r.get('split',''),r.get('growth_condition',''))
    def pick(pool,count):
        buckets=defaultdict(list)
        for pid in sorted(pool):buckets[stratum(pid)].append(pid)
        keys=sorted(buckets);rng.shuffle(keys)
        for values in buckets.values():rng.shuffle(values)
        chosen=[]
        while len(chosen)<count:
            progressed=False
            for key in keys:
                if buckets[key] and len(chosen)<count:chosen.append(buckets[key].pop());progressed=True
            if not progressed:break
        return chosen
    def subset(count):
        partial={pid for pid in available if any(r.get('stage')=='partial' for r in grouped[pid])}
        pcount=min(len(partial),round(count*partial_fraction))
        chosen=pick(partial,pcount)
        control=available-partial
        chosen+=pick(control,min(len(control),count-len(chosen)))
        chosen+=pick(available-set(chosen),count-len(chosen))
        available.difference_update(chosen);return set(chosen)
    cal=subset(calibration);agree=subset(agreement)
    # Approximate workload: partial boundaries take longer than baseline/clean scans.
    owners={};loads=Counter();group_loads=defaultdict(Counter);specimen_counts=Counter()
    blocks=sorted(available);rng.shuffle(blocks)
    blocks.sort(key=lambda pid:sum(3 if r.get('stage')=='partial' else 1 for r in grouped[pid]),reverse=True)
    for pid in blocks:
        weight=sum(3 if r.get('stage')=='partial' else 1 for r in grouped[pid]);group=stratum(pid)
        order=['dentist_1','dentist_2'];rng.shuffle(order)
        capacity=(len(blocks)+1)//2
        order=[d for d in order if specimen_counts[d]<capacity]
        owner=min(order,key=lambda d:(loads[d],group_loads[group][d]))
        owners[pid]=owner;loads[owner]+=weight;group_loads[group][owner]+=weight;specimen_counts[owner]+=1
    assigned=[]
    for row in rows:
        pid=row['physical_id'];phase='calibration' if pid in cal else 'agreement' if pid in agree else 'production'
        assigned.append({**row,'study_phase':phase,'assigned_to':'both' if phase!='production' else owners[pid]})
    return assigned


def create_plan(root,output=None,calibration=4,agreement=8,seed=42,partial_fraction=.75):
    root=Path(root).resolve();output=Path(output or root/'labeling_plan').resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('Plan output must be new/empty. Do not reshuffle cases after labeling has started.')
    rows=grouped_plan(inventory(root),calibration,agreement,seed,partial_fraction)
    output.mkdir(parents=True,exist_ok=True);write_csv(output/'assignments.csv',rows)
    queues={}
    for dentist in ('dentist_1','dentist_2'):
        for phase in ('calibration','agreement','production'):
            queues[f'{phase}_{dentist}']=[r for r in rows if r['study_phase']==phase and r['assigned_to'] in ('both',dentist)]
        queues[f'all_{dentist}']=[r for r in rows if r['assigned_to'] in ('both',dentist)]
    queues['consensus']=[r for r in rows if r['study_phase']=='agreement']
    summary={}
    for name,queue in queues.items():
        write_csv(output/(name+'.csv'),queue)
        summary[name]={'scans':len(queue),'specimens':len({r['physical_id'] for r in queue})}
        annotator='consensus' if name=='consensus' else ('dentist_1' if name.endswith('dentist_1') else 'dentist_2')
        import os
        def quote(path):return str(path).replace('%','%%')
        root_relative=quote(os.path.relpath(root,output))
        try:
            app_relative=quote(os.path.relpath(Path(__file__).parent.resolve(),output))
        except ValueError:
            app_relative=None  # Separate drives need BIOCAL3D_APP_DIR set by the user.
        locate=(f'if not defined BIOCAL3D_APP_DIR set "BIOCAL3D_APP_DIR=%~dp0{app_relative}"\r\n' if app_relative is not None else '')
        command=f'python "%BIOCAL3D_APP_DIR%\\labeling_study.py" open --root "%~dp0{root_relative}" --plan "%~dp0." --queue {name} --annotator {annotator}'
        bat=('@echo off\r\nsetlocal EnableExtensions DisableDelayedExpansion\r\n'+locate+
             'if not exist "%BIOCAL3D_APP_DIR%\\activate_biocal3d.bat" (\r\n'
             'echo Set BIOCAL3D_APP_DIR to the extracted package biocal3d_workflow folder.\r\ngoto failure\r\n)\r\n'
             'call "%BIOCAL3D_APP_DIR%\\activate_biocal3d.bat"\r\nif errorlevel 1 goto failure\r\n'+command+
             '\r\nif errorlevel 1 goto failure\r\nexit /b 0\r\n:failure\r\npause\r\nexit /b 1\r\n')
        (output/(f'open_{name}.bat')).write_bytes(bat.encode('utf-8'))
    for name in ('dentist_1','dentist_2','consensus'):(root/'annotations'/name).mkdir(parents=True,exist_ok=True)
    config={'root':str(root),'seed':seed,'calibration_specimens':calibration,'agreement_specimens':agreement,'partial_specimen_fraction':partial_fraction,
            'inventory_sha256':hashlib.sha256((root/'annotation_inventory.csv').read_bytes()).hexdigest(),
            'counts':summary,'protocol':'Calibration discussed jointly; agreement rated independently; consensus after both complete; remaining specimens assigned once for production.'}
    (output/'plan.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2));print('Plan saved:',output)
    return rows


def seed_consensus(root,plan,annotations=None):
    root=Path(root).resolve();output=Path(annotations or root/'annotations')
    if not re_valid_plan_root(Path(plan),root):raise ValueError('Plan does not match this prepared dataset.')
    with (Path(plan)/'consensus.csv').open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    stores={name:AnnotationStore(root,output,name) for name in ('dentist_1','dentist_2','consensus')}
    seeded=skipped=0
    # Validate every pair before creating any consensus draft.
    pending=[]
    for row in rows:
        mesh=load_mesh(root/row['annotation_mesh'])
        if stores['consensus'].path_for(mesh.path).exists():skipped+=1;continue
        a,adone,_=stores['dentist_1'].load(mesh);b,bdone,_=stores['dentist_2'].load(mesh)
        if not (adone and bdone):raise ValueError(f'Both dentists must complete {row.get("scan_id",mesh.path.name)} before consensus seeding.')
        pending.append((mesh,a))
    for mesh,a in pending:
        target=stores['consensus'].path_for(mesh.path).parent/(mesh.path.name+'.labeled.ply')
        export_ply(mesh,a,target)
        stores['consensus'].save(mesh,a,False,'Consensus DRAFT seeded from dentist_1. Reconcile with dentist_2; independent labels retained.')
        seeded+=1
    print(f'Seeded {seeded} consensus drafts; kept {skipped} existing consensus annotations unchanged.')
    return seeded


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    plan=sub.add_parser('plan');plan.add_argument('--root',type=Path);plan.add_argument('--output',type=Path)
    plan.add_argument('--calibration',type=int,default=4);plan.add_argument('--agreement',type=int,default=8)
    plan.add_argument('--seed',type=int,default=42);plan.add_argument('--partial-fraction',type=float,default=.75)
    folders=sub.add_parser('folders',help='Portable approved G5 calibration/agreement folders without existing plaque labels')
    folders.add_argument('--root',type=Path,required=True);folders.add_argument('--output',type=Path)
    folders.add_argument('--group',type=int,default=5);folders.add_argument('--calibration',type=int,default=2)
    folders.add_argument('--agreement',type=int,default=6);folders.add_argument('--seed',type=int,default=42)
    launch=sub.add_parser('open');launch.add_argument('--root',type=Path);launch.add_argument('--plan',type=Path)
    launch.add_argument('--queue',required=True,choices=[f'{phase}_{d}' for phase in ('calibration','agreement','production','all') for d in ('dentist_1','dentist_2')]+['consensus'])
    launch.add_argument('--annotator');launch.add_argument('--allow-rater-switch',action='store_true')
    seed=sub.add_parser('seed-consensus');seed.add_argument('--root',type=Path);seed.add_argument('--plan',type=Path);seed.add_argument('--annotations',type=Path)
    args=p.parse_args()
    try:
        root=resolve_root(args.root)
        if args.command=='folders':
            create_study_folders(root,args.output,args.group,args.calibration,args.agreement,args.seed);return 0
        if args.command=='plan':
            if not 0<=args.partial_fraction<=1:p.error('--partial-fraction must be between 0 and 1')
            create_plan(root,args.output,args.calibration,args.agreement,args.seed,args.partial_fraction);return 0
        directory=args.plan or root/'labeling_plan'
        if args.command=='seed-consensus':seed_consensus(root,directory,args.annotations);return 0
        if not re_valid_plan_root(directory,root):raise ValueError('Plan belongs to another snapshot; use its original root.')
        manifest=directory/(args.queue+'.csv');manifest_queue(root,manifest)
        entry=Path(__file__).parent/'annotator'/'biocal3d_annotator.py'
        name=args.annotator or ('consensus' if args.queue=='consensus' else ('dentist_2' if args.queue.endswith('dentist_2') else 'dentist_1'))
        command=[sys.executable,str(entry),'--root',str(root),'--manifest',str(manifest),'--annotator',name]
        if args.queue=='consensus':
            for source in manifest_queue(root,manifest):
                mesh=load_mesh(source)
                for rater in ('dentist_1','dentist_2'):
                    if not AnnotationStore(root,root/'annotations',rater).load(mesh)[1]:raise ValueError('Complete both independent ratings before opening consensus.')
            command+=['--review-mode']
        elif not args.allow_rater_switch:command+=['--lock-annotator']
        return subprocess.call(command)
    except Exception as exc:print('ERROR:',exc,file=sys.stderr);return 2


def re_valid_plan_root(directory,root):
    config=json.loads((directory/'plan.json').read_text())
    return Path(config['root']).resolve()==root.resolve() or config.get('inventory_sha256')==hashlib.sha256((root/'annotation_inventory.csv').read_bytes()).hexdigest()

if __name__=='__main__':raise SystemExit(main())
