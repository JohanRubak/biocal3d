"""Audit the four local BioCal3D dataset variants and write a scan inventory."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'studio' / 'app'))
import roi_browser_core as core
from browser_original_models import OriginalModels, sample_metadata
from original_models import OriginalModelIndex

from dataset_paths import RELEASE_FOLDERS as VARIANTS, data_root


def roi_inventory(root, progress=None):
    _, scans, warnings = core.discover(root, progress=progress)
    for warning in warnings:
        print(f'{root.name}: {warning}', file=sys.stderr)
    return {(scan.scanner, scan.sample_name): scan for scan in scans}, warnings


def source_provenance(scan):
    metadata = scan.path.with_name('metadata.json')
    fields = json.loads(metadata.read_text(encoding='utf-8-sig')) if metadata.is_file() else {}
    geometry = scan.path.with_name('geometry.npz')
    if geometry.is_file():
        with np.load(geometry, allow_pickle=False) as data:
            for key in ('source_ply', 'source_sha256'):
                if key in data and not fields.get(key):
                    fields[key] = str(data[key].item())
    return fields


def same_roi(raw, corrected):
    with np.load(raw.path.with_name('geometry.npz'), allow_pickle=False) as a, \
            np.load(corrected.path.with_name('geometry.npz'), allow_pickle=False) as b:
        for key in ('vertices', 'faces', 'raw_triangle_id', 'raw_vertex_id', 'original_to_canonical'):
            if key not in a and key not in b and key in ('raw_triangle_id', 'raw_vertex_id'):
                continue
            if key not in a or key not in b or not np.array_equal(a[key], b[key]):
                raise ValueError(f'{key} is missing or differs between matched ROI variants')
        transform = a['original_to_canonical']
        if (transform.shape != (4, 4) or not np.isfinite(transform).all() or
                not np.allclose(transform[3], [0, 0, 0, 1]) or
                abs(np.linalg.det(transform[:3, :3])) < 1e-12):
            raise ValueError('Invalid original_to_canonical transform')


def expected_keys():
    return {
        (scanner, f'BCG{group}T{stage}-{specimen}')
        for scanner in core.SCANNERS for group in range(1, 7)
        for stage in range(2 if group <= 2 else 3) for specimen in range(1, 9)
    }


def audit(roots, output, subset=False, *, progress=None):
    report = progress or (lambda message: None)
    report("Discovering approved ROI samples")
    raw_root, corrected_root, roi_root, corrected_roi_root = roots
    rois, roi_warnings = roi_inventory(roi_root, report)
    report("Discovering harmonized ROI samples")
    corrected_rois, corrected_warnings = roi_inventory(corrected_roi_root, report)
    raw_index = OriginalModelIndex(raw_root)
    corrected_index = OriginalModels(corrected_roi_root, corrected_root)
    expected = set(rois) | set(corrected_rois) if subset else expected_keys()
    failures = len(roi_warnings) + len(corrected_warnings)
    columns = ('scanner', 'sample', *VARIANTS, 'raw_sha256', 'harmonized_sha256', 'issues')
    with output.open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        scans = sorted(expected | set(rois) | set(corrected_rois))
        for number, (scanner, sample) in enumerate(scans, 1):
            status = f"Scan {number}/{len(scans)} — {scanner}/{sample}"
            report(status + " — reading native provenance")
            key = scanner, sample
            row = {'scanner': scanner, 'sample': sample}
            issues = []
            scan, color_scan = rois.get(key), corrected_rois.get(key)
            if key not in expected:
                issues.append('Unexpected scan identity for 512-scan study')
            for variant, roi, root in ((VARIANTS[2], scan, roi_root), (VARIANTS[3], color_scan, corrected_roi_root)):
                if roi is None:
                    issues.append(f'{variant} missing')
                else:
                    row[variant] = roi.path.relative_to(root).as_posix()
            if scan is not None:
                try:
                    fields = source_provenance(scan)
                    row['raw_sha256'] = fields.get('source_sha256', '')
                    if not row['raw_sha256']:
                        raise ValueError('ROI has no recorded native source SHA-256')
                    report(status + " — locating and hashing native full model")
                    raw = raw_index.resolve([fields.get('source_ply'), fields.get('source')], scanner=scanner,
                                            sample=sample, checksum=row['raw_sha256'])
                    row[VARIANTS[0]] = raw.relative_to(raw_root).as_posix()
                except (KeyError, OSError, ValueError) as error:
                    issues.append(f'{VARIANTS[0]} — {error}')
            if color_scan is not None:
                try:
                    report(status + " — locating and hashing harmonized full model")
                    metadata = sample_metadata(color_scan)
                    row['harmonized_sha256'] = metadata.get('harmonized_source_sha256', '')
                    if not row['harmonized_sha256']:
                        raise ValueError('ROI has no recorded harmonized source SHA-256')
                    corrected = corrected_index.locate(color_scan, metadata)
                    row[VARIANTS[1]] = corrected.relative_to(corrected_root).as_posix()
                except (KeyError, OSError, ValueError) as error:
                    issues.append(f'{VARIANTS[1]} — {error}')
            if scan is not None and color_scan is not None:
                try:
                    report(status + " — comparing ROI geometry")
                    same_roi(scan, color_scan)
                except (KeyError, OSError, ValueError) as error:
                    issues.append(f'ROI comparison — {error}')
            if issues:
                failures += 1
            row['issues'] = ' | '.join(issues)
            writer.writerow(row)
            report(status + (" — issues found" if issues else " — verified"))
    print(f'Inventoried {len(expected)} expected scans, {failures} scan or discovery issues. Report: {output}', flush=True)
    return int(failures != 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_root', nargs='?', type=Path, default=data_root(), help='Parent containing the four standard variant folders')
    parser.add_argument('--output', type=Path, required=True, help='New CSV file outside the input datasets')
    parser.add_argument('--subset', action='store_true', help='Audit only scans present in the supplied ROI sets')
    for variant in VARIANTS:
        parser.add_argument('--' + variant.replace('_', '-'), type=Path, help=f'Override the location of {variant}')
    args = parser.parse_args()
    roots = [(getattr(args, variant) or args.data_root / variant).resolve() for variant in VARIANTS]
    if len(set(roots)) != len(roots) or any(not root.is_dir() for root in roots):
        parser.error('Supply four distinct existing dataset folders')
    output = args.output.resolve()
    if any(output.is_relative_to(root) for root in roots):
        parser.error('Choose a report path outside the input datasets')
    try:
        return audit(roots, output, args.subset)
    except (OSError, ValueError) as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
