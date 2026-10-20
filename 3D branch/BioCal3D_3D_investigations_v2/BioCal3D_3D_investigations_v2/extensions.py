"""Study extensions: paired stage progression, interior controls and run locking."""
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict
import csv, html, json, os, socket
import numpy as np
from study_data import STAGES, write_csv


@contextmanager
def output_lock(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    lock = output / '.active_run.lock'
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise RuntimeError(f'Another launcher owns {lock}. If the earlier process was killed, '
                           'confirm it has stopped, then delete that lock file and rerun.') from None
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(dict(pid=os.getpid(), host=socket.gethostname()), f)
        yield
    finally:
        lock.unlink(missing_ok=True)


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def stage_pairs(rows):
    """Expected acquisition-stage rank is a diagnostic, not a plaque estimate."""
    groups = defaultdict(dict)
    for r in rows:
        key = (r['scanner'], r['physical_id'])
        stage = STAGES[int(r['truth'])]
        if stage in groups[key]:
            raise ValueError('Duplicate scanner/specimen/stage prediction.')
        groups[key][stage] = float(r['p_partial']) + 2 * float(r['p_clean'])
    pairs = []
    for (scanner, pid), values in sorted(groups.items()):
        for before, after in [('baseline', 'partial'), ('partial', 'clean'), ('baseline', 'clean')]:
            if before not in values or after not in values:
                continue
            delta = values[after] - values[before]
            pairs.append(dict(scanner=scanner, physical_id=pid, transition=before+' to '+after,
                              before_score=values[before], after_score=values[after], delta=delta,
                              reversed=delta < -1e-8, tied=abs(delta) <= 1e-8))
    return pairs


def rebuild_extensions(output):
    from followup_report import table
    out = Path(output); runs = []; transitions = []; summary = []; perturbations = []
    for run in sorted((out/'runs').glob('*')):
        if not (run/'config.json').exists() or not (run/'status.json').exists():
            continue
        cfg = json.loads((run/'config.json').read_text())
        if json.loads((run/'status.json').read_text()).get('status') not in ['complete', 'evaluated_checkpoint']:
            continue
        part = 'val' if cfg['protocol'] == 'reference' else 'assessment'
        metrics = run/f'metrics_{part}.json'; predictions = run/f'predictions_{part}.csv'
        if not metrics.exists() or not predictions.exists():
            continue
        m = json.loads(metrics.read_text())
        info = dict(run=run.name, protocol=cfg['protocol'], tag=cfg['tag'], model=cfg['model'],
                    variant=cfg['variant'], seed=cfg['seed'], partition=part,
                    specimen_balanced_accuracy=m['specimen_balanced_accuracy'], partial_recall=m['recall'][1])
        runs.append(info)
        sensitivity=run/f'input_sensitivity_{part}.json'
        if sensitivity.exists():
            perturbations.extend(dict(**info, transform=x['transform'], changed_predictions=x['changed_predictions'],
                mean_probability_difference=x['mean_probability_difference'],perturbed_specimen_BA=x['specimen_balanced_accuracy'])
                for x in json.loads(sensitivity.read_text()))
        pairs = stage_pairs(csv_rows(predictions))
        transitions.extend(dict(**info, **pair) for pair in pairs)
        # Describe each run separately: folds and seeds are not independent scans.
        for transition in sorted({r['transition'] for r in pairs}):
            pp = [r for r in pairs if r['transition']==transition]
            by_pid = defaultdict(list)
            for r in pp:by_pid[r['physical_id']].append(r)
            summary.append(dict(**info, transition=transition, specimens=len(by_pid), scanner_pairs=len(pp),
                specimen_mean_delta=float(np.mean([np.mean([r['delta'] for r in rows]) for rows in by_pid.values()])),
                specimen_mean_reversal_rate=float(np.mean([np.mean([r['reversed'] for r in rows]) for rows in by_pid.values()])),
                tied_pairs=sum(r['tied'] for r in pp)))
    write_csv(out/'stage_pairs.csv', transitions); write_csv(out/'stage_order_summary.csv', summary)
    write_csv(out/'input_sensitivity.csv', perturbations)
    interior = []
    for r in runs:
        if not r['tag'].startswith('interior'):continue
        full = [x for x in runs if x['protocol']=='reference' and x['tag'] in ['followup','mature'] and
                all(x[k]==r[k] for k in ['model','variant','seed'])]
        if len(full)>1:raise ValueError('Ambiguous interior/full matched comparison.')
        interior.append(dict(model=r['model'], variant=r['variant'], seed=r['seed'],
            interior_specimen_BA=r['specimen_balanced_accuracy'],
            full_specimen_BA=full[0]['specimen_balanced_accuracy'] if full else None,
            delta_interior_minus_full=r['specimen_balanced_accuracy']-full[0]['specimen_balanced_accuracy'] if full else None,
            matched_full_available=bool(full)))
    write_csv(out/'interior_comparison.csv', interior)
    baseline = [r for r in runs if r['protocol']=='reference' and r['tag']=='fixed']
    content = '''<!doctype html><html><head><meta charset="utf-8"><title>BioCal3D approved ROI investigations</title>
    <style>body{font:15px/1.55 system-ui;max-width:1300px;margin:32px auto;padding:24px;color:#203a4e}table{border-collapse:collapse;font-size:12px}td,th{padding:8px;border-bottom:1px solid #dde5eb;text-align:left}.scroll{overflow:auto}img{max-width:100%}summary{padding:12px;background:#edf3f6;cursor:pointer}.note{background:#fff4da;padding:18px}</style></head><body>
    <h1>BioCal3D · approved ROI investigations</h1>
    <p class="note">Acquisition-stage classification. All results below use development specimens; reserved test remains locked.
    Validation chooses checkpoints. Specimen CV and scanner holdout provide outer assessment.</p>
    <p><a href="followup_comparison.html">Colour pathways, CV, scanner holdout, nuisance controls and QC</a> ·
    <a href="scan_manifest.csv">Cohort / split</a> · <a href="cohort_audit.json">Approved ROI audit</a></p>'''
    for action in ['baseline','colour','interior','cv','loso']:
        p=out/f'plan_{action}.csv'
        if p.exists():content+=f'<p>{action}: {len(csv_rows(p))} planned jobs · <a href="{p.name}">plan</a></p>'
    content+='<h2>Original model and geometry investigations</h2>'+table(baseline,['model','variant','seed','specimen_balanced_accuracy','partial_recall'])
    content+='<h2>Interior ROI: matched retraining</h2><p>Retains only triangles whose three vertices lie inside 80% of the approved extraction radius by default. Full and interior models are trained independently with matching seeds and architecture. The removed annulus can contain real plaque as well as crop artifacts: a performance change alone does not identify the cause. Surface normals and centering retain the full approved-mesh convention. No remeshing or source edits.</p>'
    content+=table(interior,['model','variant','seed','matched_full_available','full_specimen_BA','interior_specimen_BA','delta_interior_minus_full'])
    content+='<h2>Within-specimen stage progression</h2><p>Stage score = P(partial) + 2 × P(clean). A negative later-minus-earlier change is a reversal of expected acquisition-stage order. This score is not plaque coverage or removal percentage. Rates average scanner pairs within each specimen first; seeds and folds remain separate.</p>'
    content+=table(summary,['run','partition','transition','specimens','scanner_pairs','specimen_mean_delta','specimen_mean_reversal_rate','tied_pairs'])
    content+='<h2>Input sensitivity of newly trained checkpoints</h2><p>Rotation/translation checks are retained for each fit. First-seed reference RGB fits also test 180°/270° rotation, point ordering, shuffled RGB and grayscale inference. These inference perturbations can create distribution shift; use the separately retrained colour pathways for learning conclusions.</p>'
    content+=table(perturbations,['run','partition','transform','changed_predictions','mean_probability_difference','perturbed_specimen_BA'])
    content+='<h2>Model focus alongside actual scanner colour</h2><p>Validation examples from first-seed reference geometry and RGB models. Predicted-class occlusion sensitivity is not a segmentation label.</p>'
    for r in runs:
        folder=out/'runs'/r['run']/'focus'; images=sorted(folder.glob('*.png'))
        if images:
            content+='<details><summary>'+html.escape(r['run'])+'</summary>'
            for p in images:content+=f'<img loading="lazy" src="{p.relative_to(out).as_posix()}">'
            content+='</details>'
    content+='<p><a href="stage_pairs.csv">Paired stage scores</a> · <a href="stage_order_summary.csv">Stage-order summary</a> · <a href="interior_comparison.csv">Interior comparisons</a></p></body></html>'
    (out/'comparison.html').write_text(content, encoding='utf-8')
