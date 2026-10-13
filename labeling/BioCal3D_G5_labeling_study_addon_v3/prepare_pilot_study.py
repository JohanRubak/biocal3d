"""Whole-specimen G5-focused study queues: keep all stages/scanners together."""
from __future__ import annotations
import argparse,csv,hashlib,json,random,re,sys
from collections import defaultdict,Counter
from pathlib import Path

RATERS=('bruna','katrin')
PROFILES=(*RATERS,'johan')


def write_csv(path,rows,empty_fields=('annotation_mesh',)):
    fields=list(dict.fromkeys(k for row in rows for k in row)) or list(empty_fields)
    with Path(path).open('w',newline='',encoding='utf-8-sig') as out:
        writer=csv.DictWriter(out,fields);writer.writeheader();writer.writerows(rows)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def group_number(row):
    value=str(row.get('group','')).upper().removeprefix('G')
    if value.isdigit():return int(value)
    match=re.match(r'G(\d+)_',row.get('physical_id',''),re.I)
    if not match:raise ValueError(f'Unknown group: {row.get("physical_id")}')
    return int(match[1])


def load_inventory(root):
    with (root/'annotation_inventory.csv').open(encoding='utf-8-sig',newline='') as src:rows=list(csv.DictReader(src))
    if not rows:raise ValueError('No approved scans in annotation_inventory.csv.')
    seen=set();keys=set()
    for row in rows:
        if not row.get('physical_id') or not row.get('scanner') or not row.get('stage'):
            raise ValueError('Inventory needs physical_id, scanner and stage columns.')
        if not row.get('annotation_mesh'):
            folder=root/row.get('sample_dir','')
            mesh=next((folder/n for n in ('roi.ply','roi.vtp') if (folder/n).is_file()),None)
            if mesh is None:raise ValueError(f'No ROI for {row.get("scan_id")}')
            row['annotation_mesh']=str(mesh.relative_to(root))
        path=(root/row['annotation_mesh']).resolve()
        if not path.is_relative_to(root) or not path.is_file():raise ValueError(f'Invalid frozen input: {path}')
        key=(row['physical_id'],row['scanner'],row.get('time_idx',row.get('timepoint',row['stage'])))
        if path in seen or key in keys:raise ValueError('Duplicate scan/specimen-scanner-timepoint in inventory.')
        seen.add(path);keys.add(key)
        if path.suffix.lower()=='.ply' and row.get('ply_sha256') and digest(path)!=row['ply_sha256']:
            raise ValueError(f'PLY changed after preprocessing: {path}')
        row['group']=str(group_number(row))
        row['pilot_cohort']='G5' if row['group']=='5' else 'standard_groups'
        # This is a group-level note, not a claim that every G5 specimen was late.
        row['timing_note']='Some G5 specimens had +2h maturation; exact affected IDs/actual hours must be recorded separately.' if row['group']=='5' else ''
    return rows


def complete_scanners(values):
    """Scanners with all three mature-group stages in the approved inventory."""
    stages=[{r['scanner'] for r in values if r['stage']==stage} for stage in ('baseline','partial','clean')]
    return set.intersection(*stages)


def choose_ids(pool,count,grouped,rng):
    ids=sorted(pool);rng.shuffle(ids)
    ids.sort(key=lambda pid:len(complete_scanners(grouped[pid])),reverse=True)
    if len(ids)<count:raise ValueError(f'Need {count} eligible specimens; only {len(ids)} available.')
    return ids[:count]


def allocate_production(rows,rng):
    grouped=defaultdict(list)
    for row in rows:grouped[row['physical_id']].append(row)
    ids=sorted(grouped);rng.shuffle(ids)
    weight=lambda pid:sum(3 if r['stage']=='partial' else 1 for r in grouped[pid])
    ids.sort(key=weight,reverse=True);owners={};loads=Counter();counts=Counter();group_load=defaultdict(Counter)
    capacity=(len(ids)+1)//2
    for pid in ids:
        group=group_number(grouped[pid][0]);order=list(RATERS);rng.shuffle(order)
        order=[n for n in order if counts[n]<capacity]
        name=min(order,key=lambda n:(loads[n],group_load[group][n]))
        owners[pid]=name;loads[name]+=weight(pid);counts[name]+=1;group_load[group][name]+=weight(pid)
    return owners


def prepare(root,output=None,calibration=2,agreement=6,agreement_g5=None,controls=0,min_scanners=4,seed=42,calibration_ids=None,agreement_ids=None,app_launcher=None):
    root=Path(root).resolve();output=Path(output or root/'labeling_g5_pilot').resolve()
    if output.is_relative_to(root/'annotations'):raise ValueError('Plan directory must be outside working annotations.')
    if output.exists() and any(output.iterdir()):raise ValueError('Plan output must be new/empty; do not reshuffle a study already in progress.')
    if min_scanners<1:raise ValueError('Positive scanner minimum required.')
    if controls!=0:raise ValueError('All stages now supply calibration controls automatically. Remove --controls; reserve whole specimens.')
    rows=load_inventory(root);grouped=defaultdict(list)
    for row in rows:grouped[row['physical_id']].append(row)
    for pid,values in grouped.items():
        if len({group_number(r) for r in values})!=1:raise ValueError(f'Conflicting group metadata for {pid}')
    eligible={pid for pid,values in grouped.items() if len(complete_scanners(values))>=min_scanners}
    g5={pid for pid in eligible if group_number(grouped[pid][0])==5}
    standard={pid for pid in eligible if group_number(grouped[pid][0]) in (3,4,6)}
    calibration=calibration if calibration is not None else len(calibration_ids) if calibration_ids else 2
    agreement=agreement if agreement is not None else len(agreement_ids) if agreement_ids else 6
    agreement_g5=agreement if agreement_g5 is None else agreement_g5
    if calibration<1 or agreement<2 or not 0<=agreement_g5<=agreement:
        raise ValueError('Invalid counts: positive calibration, agreement >=2, G5 count within agreement.')
    rng=random.Random(seed)
    def explicit(values,pool,count,label):
        if len(values)!=count or len(set(values))!=count or not set(values)<=pool:
            raise ValueError(f'{label} IDs must be {count} distinct eligible specimens in the required groups, with at least {min_scanners} scanners complete across baseline/partial/clean.')
        return list(values)
    cal=explicit(calibration_ids,g5,calibration,'Calibration') if calibration_ids else choose_ids(g5,calibration,grouped,rng)
    if agreement_ids:
        agree=explicit(agreement_ids,eligible-set(cal),agreement,'Agreement')
        if sum(pid in g5 for pid in agree)!=agreement_g5 or not set(agree)<=g5|standard:raise ValueError('Agreement IDs must match the G5/standard-group counts.')
    else:
        agree=choose_ids(g5-set(cal),agreement_g5,grouped,rng)
        # Include G6 when possible and a second standard group, then fill evenly.
        buckets=defaultdict(set)
        for pid in standard:buckets[group_number(grouped[pid][0])].add(pid)
        order=sorted(buckets);rng.shuffle(order)
        if 6 in order:order.remove(6);order.insert(0,6)
        needed=agreement-agreement_g5;chosen=[]
        while len(chosen)<needed:
            advanced=False
            for group in order:
                if buckets[group] and len(chosen)<needed:
                    pid=choose_ids(buckets[group],1,grouped,rng)[0];buckets[group].remove(pid);chosen.append(pid);advanced=True
            if not advanced:raise ValueError(f'Not enough standard-group specimens for {needed} agreement specimens.')
        agree+=chosen
    pilot_cal=[r for r in rows if r['physical_id'] in cal]
    pilot_agree=[r for r in rows if r['physical_id'] in agree]
    selected_cal={r['annotation_mesh'] for r in pilot_cal};selected_agree={r['annotation_mesh'] for r in pilot_agree}
    production=[r for r in rows if r['annotation_mesh'] not in selected_cal|selected_agree]
    owners=allocate_production(production,rng)
    assigned=[]
    for row in rows:
        path=row['annotation_mesh'];phase='calibration' if path in selected_cal else 'agreement' if path in selected_agree else 'production'
        assigned.append({**row,'study_phase':phase,'assigned_to':'both' if phase!='production' else owners[row['physical_id']]})
    launcher=Path(app_launcher or Path(__file__).parent/'annotator'/'start_annotator.bat').resolve()
    if not launcher.is_file():raise ValueError('Cannot find annotator launcher. Put this helper in biocal3d_workflow, or pass --app-launcher with the existing start_annotator.bat path.')
    output.mkdir(parents=True,exist_ok=True)
    write_csv(output/'assignments.csv',assigned)
    queues={}
    for name in RATERS:
        for phase in ('calibration','agreement','production'):
            queues[f'{phase}_{name}']=[r for r in assigned if r['study_phase']==phase and r['assigned_to'] in ('both',name)]
        queues[f'all_{name}']=[r for r in assigned if r['assigned_to'] in ('both',name)]
    queues['experiment_johan']=[{**r,'assigned_to':'johan_experiment'} for r in assigned]
    counts={}
    def quote(value):return str(value).replace('%','%%')
    for key,values in queues.items():
        name='johan' if key=='experiment_johan' else key.rsplit('_',1)[1]
        manifest=output/(key+'.csv');write_csv(manifest,values)
        counts[key]={'scans':len(values),'specimens':len({r['physical_id'] for r in values})}
        locked=' --lock-annotator'
        command=f'call "{quote(launcher)}" --root "{quote(root)}" --manifest "{quote(manifest)}" --annotator {name}{locked}'
        (output/(f'open_{key}.bat')).write_bytes(('@echo off\r\nsetlocal EnableExtensions DisableDelayedExpansion\r\n'+command+'\r\nexit /b %ERRORLEVEL%\r\n').encode('utf-8'))
    for name in PROFILES:(root/'annotations'/name).mkdir(parents=True,exist_ok=True)
    timing=[{'physical_id':pid,'group':group_number(grouped[pid][0]),'expected_growth_hours':12 if group_number(grouped[pid][0])<3 else 24,
             'actual_growth_hours':'','timing_status':'needs_recording' if group_number(grouped[pid][0])==5 else '','timing_notes':''} for pid in sorted(grouped)]
    write_csv(output/'specimen_timing_to_complete.csv',timing)
    config={'root':str(root),'inventory_sha256':digest(root/'annotation_inventory.csv'),'raters':list(RATERS),'experiment_profile':'johan',
            'agreement_manifest':'agreement_bruna.csv','calibration_ids':cal,'agreement_ids':agree,'seed':seed,'counts':counts,
            'schema_version':2,'subset_unit':'physical_specimen_all_stages_all_scanners',
            'sample_size_status':'all_eligible_standard_specimens_precision_not_guaranteed' if agreement_g5==0 and set(agree)==standard else 'pilot_not_precision_certified',
            'pilot_scope':'G5_only' if all(pid in g5 for pid in agree) else 'standard_groups' if agreement_g5==0 else 'mixed_groups',
            'production_status':'reserved_until_pilot_review',
            'precision_target':{'metric':'mean specimen-level partial-stage plaque Dice','confidence':0.95,'half_width':0.05,'recommended_primary_cohort':'G5' if all(pid in g5 for pid in agree) else 'standard_groups','intended_future_cohort':'standard_groups'},
            'pilot_decision':'Review independent G5 agreement, uncertainty coverage and between-specimen variation before deciding the next labeling stage. G5 results do not certify standard-group precision.',
            'eligible_agreement_specimens_after_calibration':{'G5':len(g5-set(cal)),'standard_groups':len(standard),'all':len((g5|standard)-set(cal))},
            'agreement_specimens_by_cohort':{'G5':sum(pid in g5 for pid in agree),'standard_groups':sum(pid in standard for pid in agree)},
            'consensus_required':False,'metadata_blinded':False,'independent_ratings_required':True,'primary_stage':'partial',
            'timing_policy':'G5 is an extra group with specimen-specific timing to document; group alone does not establish actual growth time.',
            'protocol':'Keep every available stage/scanner of a specimen in the same study subset. Discuss calibration jointly; freeze rules; rate agreement independently with scanner/stage/group visible; compare before any reconciliation. Johan excluded from agreement.'}
    (output/'plan.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    analyser=Path(__file__).parent/'analyse_labeling_agreement.py'
    prefix='@echo off\r\nsetlocal EnableExtensions DisableDelayedExpansion\r\nif not defined BIOCAL3D_STUDY_PYTHON set "BIOCAL3D_STUDY_PYTHON=%LOCALAPPDATA%\\anaconda3\\envs\\biocal3d\\python.exe"\r\nif not exist "%BIOCAL3D_STUDY_PYTHON%" goto failure\r\n'
    analysis_bat=prefix+f'"%BIOCAL3D_STUDY_PYTHON%" "{quote(analyser)}" --root "{quote(root)}" --plan "{quote(output)}"\r\nset "BIOCAL3D_EXIT_CODE=%ERRORLEVEL%"\r\npause\r\nexit /b %BIOCAL3D_EXIT_CODE%\r\n:failure\r\necho Set BIOCAL3D_STUDY_PYTHON to the biocal3d python.exe path.\r\npause\r\nexit /b 1\r\n'
    (output/'analyse_agreement.bat').write_bytes(analysis_bat.encode('utf-8'))
    print(json.dumps(config,indent=2));print('Prepared pilot:',output)
    print('Inspect assignments and QC before freezing the queues. Fill specimen_timing_to_complete.csv; no G5 specimen was automatically marked +2h.')
    return assigned,config


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--calibration',type=int,default=2,help='G5 calibration specimens; default 2, all stages/scanners.')
    p.add_argument('--agreement',type=int,default=6,help='Agreement specimens; default 6, all stages/scanners.')
    p.add_argument('--agreement-g5',type=int,help='G5 specimens within agreement; default all agreement specimens. Explicitly reduce to include G3/G4/G6.')
    p.add_argument('--controls',type=int,default=0,help=argparse.SUPPRESS);p.add_argument('--min-scanners',type=int,default=4);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--calibration-ids',nargs='+');p.add_argument('--agreement-ids',nargs='+');p.add_argument('--app-launcher',type=Path)
    args=p.parse_args()
    try:
        prepare(args.root,args.output,args.calibration,args.agreement,args.agreement_g5,args.controls,args.min_scanners,args.seed,args.calibration_ids,args.agreement_ids,args.app_launcher)
        return 0
    except Exception as exc:print('ERROR:',exc,file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
