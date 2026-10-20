"""Specimen-grouped calibration, independent agreement, consensus and production queues."""
from __future__ import annotations
import argparse,csv,json,random,subprocess,sys,hashlib
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from annotator.mesh_io import AnnotationStore,load_mesh,manifest_queue,export_ply


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
    launch=sub.add_parser('open');launch.add_argument('--root',type=Path);launch.add_argument('--plan',type=Path)
    launch.add_argument('--queue',required=True,choices=[f'{phase}_{d}' for phase in ('calibration','agreement','production','all') for d in ('dentist_1','dentist_2')]+['consensus'])
    launch.add_argument('--annotator');launch.add_argument('--allow-rater-switch',action='store_true')
    seed=sub.add_parser('seed-consensus');seed.add_argument('--root',type=Path);seed.add_argument('--plan',type=Path);seed.add_argument('--annotations',type=Path)
    args=p.parse_args()
    try:
        root=resolve_root(args.root)
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
