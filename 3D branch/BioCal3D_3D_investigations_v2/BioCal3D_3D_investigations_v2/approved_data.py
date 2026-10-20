"""Portable, read-only validation of complete rebuilt approved-ROI snapshots."""
from pathlib import Path
import json
import numpy as np
from study_data import load_snapshot, reference_split, digest, default_snapshot_root


def default_root():
    return default_snapshot_root()


def load_approved(args):
    root = args.data_root.expanduser().resolve()
    if root.is_file():
        root = root.parent
    if (root / 'INCOMPLETE_REBUILD.json').exists():
        raise ValueError('INCOMPLETE_REBUILD.json exists. Finish rebuilding into a new empty folder before training.')
    marker = root / 'geometry_rebuild.json'
    index = root / 'dataset_index.json'
    if not marker.is_file() or not index.is_file():
        raise ValueError(f'{root}: expected the COMPLETE rebuilt snapshot, including dataset_index.json, '
                         'geometry_rebuild.json and samples/.../geometry.npz. Copy the whole Approved ROIs '
                         'folder, not only annotations/_roi_review. Use --data-root if it has another name.')
    rebuild = json.loads(marker.read_text(encoding='utf-8-sig'))
    if rebuild.get('status') != 'complete':
        raise ValueError('ROI rebuild is not marked complete.')
    doc = json.loads(index.read_text(encoding='utf-8-sig'))
    source = doc.get('scans', [])
    if not source or any(str(r.get('geometry_review_status', '')).casefold() != 'approved' for r in source):
        raise ValueError('Every scan in dataset_index.json must have geometry_review_status=approved.')
    reviewed = rebuild.get('scans', [])
    records = {r.get('scan_id'): r for r in source}
    review_ids = [r.get('scan_id') for r in reviewed]
    if len(set(review_ids)) != len(review_ids) or set(review_ids) != set(records):
        raise ValueError('Rebuild approval inventory does not match dataset_index.json.')
    if any(str(r.get('status', '')).casefold() != 'approved' for r in reviewed):
        raise ValueError('Rebuild inventory contains unapproved ROIs.')
    # Complete experimental cohort is required; no silent CHECK/status exclusions.
    raw, audit, excluded = load_snapshot(root, allow_incomplete=False)
    if excluded:
        raise ValueError('Approved snapshot contains non-OK scans. Review preparation before training.')
    if any(not r['hash_supplied'] for r in raw):
        raise ValueError('Rebuilt snapshots must provide geometry_sha256 for every scan.')
    if args.split_json:
        rows = reference_split(raw, args.split_seed, args.split_json)
        policy = 'explicit split JSON'
    elif args.split_policy == 'source':
        if not all(r['source_split'] for r in raw):
            raise ValueError('--split-policy source requires a complete saved source split.')
        rows = reference_split(raw, args.split_seed)
        policy = 'preserved source split'
    else:
        # Do not inherit a prior run partition merely because the rebuild copied it.
        rows = reference_split([dict(r, source_split='') for r in raw], args.split_seed)
        policy = 'new grouped split from this approved cohort'
    audit.update(approved_scans=len(rows), rebuild_sha256=digest(marker), split_policy=policy,
                 split_seed=args.split_seed, test_policy='identity/hash only; no test geometry arrays, QC or predictions')
    return rows, audit, excluded


def interior_mask(row, vertices, faces, fraction):
    """All three triangle vertices inside a smaller reviewed extraction circle.

    Uses the approved extraction's actual center/tilt, not a refitted circle.
    Source triangle indexing is preserved for subsequent focus maps.
    """
    if not 0 < fraction <= 1:
        raise ValueError('Interior fraction must be in (0,1].')
    with np.load(row['mesh_path'], allow_pickle=False) as z:
        extraction = json.loads(str(z['extraction_json'].item())) if 'extraction_json' in z else {}
    settings = extraction.get('settings')
    if not isinstance(settings, dict) or 'radius' not in settings:
        raise ValueError(f'{row["scan_id"]}: missing reviewed extraction settings for interior ROI.')
    radius = float(settings['radius'])
    tx, ty = np.deg2rad([float(settings.get('tilt_x', 0)), float(settings.get('tilt_y', 0))])
    center = np.array([float(settings.get(k, 0)) for k in ['cx', 'cy', 'z']])
    normal = np.array([np.tan(ty), -np.tan(tx), 1.])
    if not np.isfinite([*center, *normal, radius]).all() or radius <= 0:
        raise ValueError('Invalid reviewed circle settings.')
    normal /= np.linalg.norm(normal)
    u = np.cross([0., 1., 0.], normal); u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    relative = vertices - center
    radial_sq = (relative @ u)**2 + (relative @ v)**2
    return (radial_sq <= (fraction * radius)**2 + 1e-10)[faces].all(axis=1)
