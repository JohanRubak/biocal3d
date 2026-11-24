"""Delete only reviewed obsolete dataset files after a complete upload verification."""
import argparse
import csv
import json
from pathlib import Path, PurePosixPath
import re

from audit_datasets import VARIANTS, expected_keys


def safe_path(value):
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or '\\' in value or value != path.as_posix()
            or any(part in ('.', '..') for part in value.split('/'))
            or any(char in value for char in '*?[]\n\r') or ':' in value):
        raise ValueError(f'Unsafe repository file path: {value!r}')
    return value


def load_verification(path, revision):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'path', 'expected_bytes', 'expected_source_sha256', 'remote_sha256', 'status', 'revision'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('The upload verification report is missing required columns.')
        rows = list(reader)
    if not rows or any(row['revision'] != revision or row['status'] != 'verified' for row in rows):
        raise ValueError('Every report entry must be verified at the specified dataset revision.')
    protected = {}
    for row in rows:
        name = safe_path(row['path'])
        if name in protected or int(row['expected_bytes']) < 0:
            raise ValueError('The verification report contains duplicate paths or invalid sizes.')
        protected[name] = row
    if not {'README.md', 'dataset_inventory.csv'}.issubset(protected):
        raise ValueError('The report must protect the dataset card and inventory.')
    for variant in VARIANTS:
        identities = set()
        for name, row in protected.items():
            parts = PurePosixPath(name).parts
            if parts[0] != variant:
                continue
            is_full = variant in VARIANTS[:2]
            if is_full and not row['expected_source_sha256']:
                continue
            if not is_full and parts[-1] != 'geometry.npz':
                continue
            if is_full and not re.fullmatch(r'[0-9a-fA-F]{64}', row['expected_source_sha256']):
                raise ValueError('A full model has invalid source-checksum evidence.')
            matches = set(re.findall(r'BCG[1-6]T[012]-[1-8](?!\d)', name))
            if len(parts) < 3 or len(matches) != 1:
                raise ValueError(f'Cannot identify the protected study scan: {name}')
            key = parts[1], matches.pop()
            if key in identities:
                raise ValueError(f'Duplicate protected study scan in {variant}: {key}')
            identities.add(key)
        if identities != expected_keys():
            raise ValueError(f'The report does not protect all 512 unique study scans in {variant}.')
    return protected


def validate_remote(metadata, protected):
    if not metadata.private:
        raise ValueError('Cleanup requires the dataset to remain private.')
    remote = {entry.rfilename: entry for entry in metadata.siblings}
    for name, row in protected.items():
        entry = remote.get(name)
        if entry is None or entry.size != int(row['expected_bytes']):
            raise ValueError(f'A protected file is missing or has a different size: {name}')
        expected = row['expected_source_sha256'] or row['remote_sha256']
        if expected:
            lfs = getattr(entry, 'lfs', None)
            checksum = lfs.get('sha256', '') if isinstance(lfs, dict) else getattr(lfs, 'sha256', '')
            if checksum.lower() != expected.lower():
                raise ValueError(f'A protected file checksum does not match the verified upload: {name}')
    return remote


def cleanup(api, repo_id, report, obsolete_file, revision, *, apply=False):
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Use the full 40-character verified dataset revision.')
    protected = load_verification(report, revision)
    lines = Path(obsolete_file).read_text(encoding='utf-8-sig').splitlines()
    obsolete = [safe_path(line.strip()) for line in lines if line.strip()]
    if not obsolete or len(obsolete) != len(set(obsolete)):
        raise ValueError('The obsolete-path list must contain unique file paths.')
    for name in obsolete:
        if PurePosixPath(name).parts[0] not in VARIANTS or name in protected:
            raise ValueError(f'Refusing to delete a protected or non-data path: {name}')
    head = api.dataset_info(repo_id)
    if head.sha != revision:
        raise ValueError(f'Dataset changed since verification: current revision is {head.sha}. '
                         'Do not use the old cleanup list; verify the current release again.')
    metadata = api.dataset_info(repo_id, revision=revision, files_metadata=True)
    remote = validate_remote(metadata, protected)
    actual_obsolete = {name for name in remote if name.split('/')[0] in VARIANTS and name not in protected}
    if set(obsolete) != actual_obsolete:
        raise ValueError('The obsolete-path list does not exactly match unneeded data files at this revision.')
    print(f'Protected verified files: {len(protected):,}', flush=True)
    print(f'Obsolete files to remove: {len(obsolete):,}', flush=True)
    plan = Path(report).parent / f'cleanup-plan-{revision}.json'
    plan.write_text(json.dumps({'repo_id': repo_id, 'revision_before': revision,
                               'verification_report': str(report), 'obsolete_files': obsolete,
                               'protected_count': len(protected)}, indent=2) + '\n', encoding='utf-8')
    print(f'Exact cleanup plan: {plan}', flush=True)
    if not apply:
        print('Preview only. No remote files changed. Add --apply to remove these obsolete files.', flush=True)
        return None
    from huggingface_hub import CommitOperationDelete
    print('Removing obsolete files in one revision-guarded commit...', flush=True)
    result = api.create_commit(repo_id=repo_id, repo_type='dataset', revision='main',
                               parent_commit=revision,
                               operations=[CommitOperationDelete(path_in_repo=name, is_folder=False) for name in obsolete],
                               commit_message=f'Remove {len(obsolete)} obsolete BioCal3D data paths')
    after = api.dataset_info(repo_id, revision=result.oid, files_metadata=True)
    remaining = validate_remote(after, protected)
    if set(remaining) != set(remote) - set(obsolete):
        raise ValueError(f'Cleanup commit {result.oid} needs review: unexpected final file list.')
    completion = plan.with_name(f'cleanup-completed-{result.oid}.json')
    completion.write_text(json.dumps({'repo_id': repo_id, 'revision_before': revision,
                                     'revision_after': result.oid, 'removed_files': obsolete,
                                     'protected_files_verified': len(protected)}, indent=2) + '\n', encoding='utf-8')
    print(f'Cleanup verified: removed {len(obsolete)} obsolete files; retained all {len(protected):,} verified files.', flush=True)
    print(f'Dataset revision: {result.oid}', flush=True)
    print(f'Cleanup record: {completion}', flush=True)
    print('Local data was not changed. Earlier remote filenames remain available in dataset history.', flush=True)
    return result.oid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--verification-report', required=True, type=Path)
    parser.add_argument('--obsolete-paths', required=True, type=Path)
    parser.add_argument('--expected-revision', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    from huggingface_hub import HfApi
    cleanup(HfApi(), args.repo_id, args.verification_report, args.obsolete_paths,
            args.expected_revision, apply=args.apply)


if __name__ == '__main__':
    main()
