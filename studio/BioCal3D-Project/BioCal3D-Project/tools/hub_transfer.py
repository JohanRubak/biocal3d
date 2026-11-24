"""Binary-safe dataset transfer and revision-specific completeness checks."""
import csv
from datetime import datetime
from pathlib import Path

from audit_datasets import VARIANTS

BINARY_EXTENSIONS = ('ply', 'vtp', 'npz', 'npy', 'png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp', 'zip')


def upload_study_archive(api, repo_id, archive, manifest):
    """Publish the supplied final-run ZIP separately from the reference scans."""
    import hashlib
    import json
    from pathlib import PurePosixPath
    archive, manifest = Path(archive), Path(manifest)
    record = json.loads(manifest.read_text(encoding='utf-8'))
    if archive.stat().st_size != record['archive_bytes']:
        raise ValueError('The run ZIP size does not match the inspected archive manifest.')
    print('Checking classification run archive checksum...', flush=True)
    digest = hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != record['archive_sha256']:
        raise ValueError('The run ZIP checksum does not match the inspected archive manifest.')
    paths = [record['artifact_path'], record['manifest_path']]
    for name in paths:
        path = PurePosixPath(name)
        if (not name.startswith('artifacts/classification/') or path.is_absolute()
                or '..' in path.parts or '\\' in name or path.as_posix() != name):
            raise ValueError('Research artifacts must stay under artifacts/classification/.')
    before = api.dataset_info(repo_id, files_metadata=True)
    if not before.private:
        raise ValueError('Research artifacts require a private dataset repository.')

    def signature(entry):
        lfs = getattr(entry, 'lfs', None)
        sha = lfs.get('sha256', '') if isinstance(lfs, dict) else getattr(lfs, 'sha256', '')
        return entry.size, entry.blob_id, sha

    protected = {entry.rfilename: signature(entry) for entry in before.siblings
                 if entry.rfilename not in (*paths, '.gitattributes')}
    for entry in before.siblings:
        if entry.rfilename == paths[0] and signature(entry)[2] != record['archive_sha256']:
            raise ValueError('A different artifact already occupies this path; choose a new archive version.')
    configure_binary_storage(api, repo_id, [entry.rfilename for entry in before.siblings])
    head = api.dataset_info(repo_id)
    print('Uploading the final classification ZIP outside the four scan variants...', flush=True)
    result = api.upload_file(repo_id=repo_id, repo_type='dataset', revision='main', parent_commit=head.sha,
                             path_or_fileobj=str(archive), path_in_repo=paths[0],
                             commit_message='Archive the final BioCal3D classification run')
    result = api.upload_file(repo_id=repo_id, repo_type='dataset', revision='main', parent_commit=result.oid,
                             path_or_fileobj=str(manifest), path_in_repo=paths[1],
                             commit_message='Record classification archive provenance')
    after = api.dataset_info(repo_id, revision=result.oid, files_metadata=True)
    remaining = {entry.rfilename: signature(entry) for entry in after.siblings
                 if entry.rfilename not in (*paths, '.gitattributes')}
    remote = {entry.rfilename: entry for entry in after.siblings}
    if not after.private or remaining != protected:
        raise ValueError('Artifact publication changed unrelated dataset metadata; review the printed commit.')
    archived = signature(remote[paths[0]]) if paths[0] in remote else None
    if archived is None or archived[0] != record['archive_bytes'] or archived[2] != record['archive_sha256']:
        raise ValueError('Uploaded run archive size/checksum could not be verified.')
    downloaded = api.hf_hub_download(repo_id=repo_id, repo_type='dataset', revision=result.oid, filename=paths[1])
    if Path(downloaded).read_bytes() != manifest.read_bytes():
        raise ValueError('The uploaded artifact manifest does not match its source.')
    print('Classification archive and manifest verified; all existing scan files are unchanged.', flush=True)
    print(f'Artifact revision: {result.oid}', flush=True)
    print(f'Archive path: {paths[0]}', flush=True)
    return result.oid


def update_readme(api, repo_id, card):
    """Publish the dataset card without touching any other repository file."""
    before = api.dataset_info(repo_id, files_metadata=True)
    if not before.private:
        raise ValueError('The target dataset must remain private.')

    def signatures(info):
        result = {}
        for entry in info.siblings:
            if entry.rfilename == 'README.md':
                continue
            lfs = getattr(entry, 'lfs', None)
            checksum = lfs.get('sha256', '') if isinstance(lfs, dict) else getattr(lfs, 'sha256', '')
            result[entry.rfilename] = (entry.size, entry.blob_id, checksum)
        return result

    protected = signatures(before)
    if not any(name.startswith(variant + '/') for name in protected for variant in VARIANTS):
        raise ValueError('The target has no prepared BioCal3D data; check the repository ID.')
    result = api.upload_file(repo_id=repo_id, repo_type='dataset', revision='main',
                             parent_commit=before.sha, path_or_fileobj=str(card),
                             path_in_repo='README.md', commit_message='Document BioCal3D data for incoming researchers')
    after = api.dataset_info(repo_id, revision=result.oid, files_metadata=True)
    if not after.private or signatures(after) != protected:
        raise ValueError(f'Documentation commit {result.oid} requires review: non-README metadata changed.')
    downloaded = api.hf_hub_download(repo_id=repo_id, repo_type='dataset',
                                     revision=result.oid, filename='README.md')
    if Path(downloaded).read_bytes() != Path(card).read_bytes():
        raise ValueError('The published dataset README does not match its source.')
    print('Dataset README updated and verified; all other repository files are unchanged.', flush=True)
    print(f'Documentation revision: {result.oid}', flush=True)
    return result.oid


def binary_attributes(existing=''):
    # Preserve repository-specific rules; put the explicit binary rules last.
    rules = '\n'.join('*.' + ''.join(f'[{c.lower()}{c.upper()}]' for c in suffix)
                      + ' filter=lfs diff=lfs merge=lfs -text' for suffix in BINARY_EXTENSIONS)
    block = '# BioCal3D binary data\n' + rules + '\n# End BioCal3D binary data'
    import re
    existing = re.sub(r'# BioCal3D binary data\n.*?# End BioCal3D binary data\n?', '', existing, flags=re.S)
    return existing.rstrip() + '\n' + block + '\n'


def configure_binary_storage(api, repo_id, existing_paths):
    existing = ''
    if '.gitattributes' in existing_paths:
        path = api.hf_hub_download(repo_id=repo_id, repo_type='dataset', filename='.gitattributes')
        existing = Path(path).read_text(encoding='utf-8')
    content = binary_attributes(existing)
    print('Configuring binary storage for PLY, VTP, NumPy arrays and textures...', flush=True)
    api.upload_file(repo_id=repo_id, repo_type='dataset', path_in_repo='.gitattributes',
                    path_or_fileobj=content.encode('utf-8'), commit_message='Configure BioCal3D binary data storage')


def check_remote(api, repo_id, expected_sizes, rows, report_folder):
    info = api.dataset_info(repo_id)
    if not info.private:
        raise ValueError('The target dataset is no longer private.')
    metadata = api.dataset_info(repo_id, revision=info.sha, files_metadata=True)
    remote = {entry.rfilename: entry for entry in metadata.siblings}
    checksums = {variant + '/' + row[variant]: row[column]
                 for row in rows for variant, column in zip(VARIANTS[:2], ('raw_sha256', 'harmonized_sha256'))}
    results = []
    for name, size in sorted(expected_sizes.items()):
        entry = remote.get(name)
        expected_sha = checksums.get(name, '')
        lfs = getattr(entry, 'lfs', None)
        remote_sha = (lfs.get('sha256', '') if isinstance(lfs, dict) else getattr(lfs, 'sha256', '')) if lfs else ''
        status = ('missing' if entry is None else 'size mismatch' if entry.size != size
                  else 'source checksum unavailable' if expected_sha and not remote_sha
                  else 'source checksum mismatch' if expected_sha and remote_sha.lower() != expected_sha.lower()
                  else 'verified')
        results.append(dict(path=name, expected_bytes=size, remote_bytes=getattr(entry, 'size', ''),
                            expected_source_sha256=expected_sha, remote_sha256=remote_sha,
                            status=status, revision=info.sha))
    report_folder = Path(report_folder)
    report_folder.mkdir(parents=True, exist_ok=True)
    report = report_folder / ('upload-verification-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '.csv')
    with report.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    for variant in VARIANTS:
        subset = [row for row in results if row['path'].startswith(variant + '/')]
        verified = sum(row['status'] == 'verified' for row in subset)
        print(f'{variant}: {verified:,}/{len(subset):,} files verified', flush=True)
    print(f'Verification report: {report}', flush=True)
    failed = [row for row in results if row['status'] != 'verified']
    if failed:
        raise ValueError(f"{len(failed)} files failed remote verification; first: {failed[0]['path']} "
                         f"({failed[0]['status']}). Rerun to complete/repair the upload. See {report}")
    stale = sorted(name for name in remote if name.split('/')[0] in VARIANTS and name not in expected_sizes)
    return info.sha, stale
