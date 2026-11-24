"""Audit and upload the BioCal3D data release to a private Hugging Face repository."""
import argparse
import csv
import hashlib
import re
import shutil
import sys
import traceback
from datetime import datetime
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from threading import Event, Thread
from time import monotonic
from pathlib import Path, PureWindowsPath
import tempfile

from audit_datasets import VARIANTS, audit, expected_keys
from prepare_data_release import release_files


@contextmanager
def progress_report(initial):
    current = [initial]
    stopped = Event()
    started = monotonic()

    def update(message):
        current[0] = message

    def heartbeat():
        while not stopped.wait(30):
            minutes = (monotonic() - started) / 60
            print(f"[{minutes:.1f} min] {current[0]}", flush=True)

    print(initial, flush=True)
    worker = Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        yield update
    finally:
        stopped.set()
        worker.join()


def validated_inventory(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'scanner', 'sample', 'issues', 'raw_sha256', 'harmonized_sha256', *VARIANTS}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('The saved audit is missing required columns.')
        rows = list(reader)
    keys = [(row['scanner'], row['sample']) for row in rows]
    if len(keys) != 512 or set(keys) != expected_keys():
        raise ValueError('The saved audit must contain all 512 unique study scans.')
    for row in rows:
        if (row['issues'] or '').strip():
            raise ValueError('The saved audit contains unresolved issues.')
        for column in ('raw_sha256', 'harmonized_sha256'):
            if not re.fullmatch(r'[0-9a-fA-F]{64}', row[column] or ''):
                raise ValueError(f'The saved audit has an invalid {column}.')
        for variant in VARIANTS:
            value = row[variant] or ''
            relative = PureWindowsPath(value)
            if not value or relative.drive or relative.root or '..' in relative.parts:
                raise ValueError(f'The saved audit has an unsafe {variant} path.')
    return rows


def upload(repo_id, data_root, dry_run=False, *, audit_inventory=None):
    data_root = data_root.expanduser().resolve()
    roots = [data_root / name for name in VARIANTS]
    with tempfile.TemporaryDirectory(prefix='biocal3d-upload-') as temporary:
        inventory = Path(temporary) / 'dataset_inventory.csv'
        if audit_inventory is not None:
            rows = validated_inventory(audit_inventory.expanduser().resolve())
            shutil.copyfile(audit_inventory.expanduser().resolve(), inventory)
            print(f"Reusing completed audit: {audit_inventory}", flush=True)
        else:
            with progress_report("Auditing local data before upload...") as report:
                audit_failed = audit(roots, inventory, progress=report)
            if audit_failed:
                raise ValueError('The release audit found issues. Run tools/audit_datasets.py to save the report.')
            rows = validated_inventory(inventory)
            saved_folder = Path.home() / 'BioCal3D' / 'upload-audits'
            saved_folder.mkdir(parents=True, exist_ok=True)
            root_key = hashlib.sha256(str(data_root).encode()).hexdigest()[:12]
            stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
            saved = saved_folder / f'{root_key}-{stamp}.csv'
            shutil.copyfile(inventory, saved)
            print(f"Completed audit saved for restart: {saved}", flush=True)
        selected = set()
        sizes = {}
        with progress_report("Selecting required data files and calculating upload size...") as report:
            for root, variant in zip(roots, VARIANTS):
                report(f"Selecting files in {variant}")
                selected.update(path.relative_to(data_root).as_posix()
                                for path in release_files(root, rows, variant, roots[2], progress=report))
            selected = sorted(selected)
            report("Reading file sizes from the data drive")
            size = 0
            for number, name in enumerate(selected, 1):
                report(f"Reading size {number}/{len(selected)} — {name}")
                sizes[name] = (data_root / name).stat().st_size
                size += sizes[name]
        print(f'Selected {len(selected):,} data files, {size / 1024**3:.2f} GiB.', flush=True)
        if dry_run:
            return
        from huggingface_hub import HfApi
        print("Connecting to Hugging Face and checking private repository...", flush=True)
        api = HfApi()
        api.create_repo(repo_id, repo_type='dataset', private=True, exist_ok=True)
        if not api.dataset_info(repo_id).private:
            raise ValueError('The target repository is public. Select a private repository for this upload.')
        print('Checking whether the remote dataset contains older folder paths...', flush=True)
        existing = api.list_repo_files(repo_id, repo_type='dataset')
        stale = sorted(name for name in existing if name.split('/')[0] in VARIANTS and name not in selected)
        if stale:
            print(f'{len(stale):,} older remote data paths will be preserved while the new release uploads.', flush=True)
        from hub_transfer import configure_binary_storage, check_remote
        configure_binary_storage(api, repo_id, existing)
        card = Path(__file__).resolve().parents[1] / 'docs' / 'dataset-card.md'
        print("Uploading dataset description and inventory...", flush=True)
        api.upload_file(repo_id=repo_id, repo_type='dataset',
                        path_or_fileobj=str(card), path_in_repo='README.md')
        api.upload_file(repo_id=repo_id, repo_type='dataset',
                        path_or_fileobj=str(inventory), path_in_repo='dataset_inventory.csv')
        for number, (root, variant) in enumerate(zip(roots, VARIANTS), 1):
            prefix = variant + '/'
            patterns = [name[len(prefix):] for name in selected if name.startswith(prefix)]
            if not patterns:
                raise ValueError(f'No selected files for {variant}; upload cannot be complete.')
            print(f'Uploading folder {number}/4: {variant} ({len(patterns):,} selected files)', flush=True)
            with progress_report(f'{variant}: Hugging Face is preparing or transferring this folder.'):
                api.upload_folder(repo_id=repo_id, repo_type='dataset', folder_path=str(root),
                                  path_in_repo=variant, allow_patterns=patterns,
                                  commit_message=f'Upload BioCal3D {variant}')
            print(f'Transfer finished for {variant}', flush=True)
        sizes['README.md'] = card.stat().st_size
        sizes['dataset_inventory.csv'] = inventory.stat().st_size
        print('Verifying all four folders, file sizes and full-model source checksums...', flush=True)
        revision, stale = check_remote(api, repo_id, sizes, rows, Path.home() / 'BioCal3D' / 'upload-reports')
        if stale:
            obsolete = Path.home() / 'BioCal3D' / 'upload-reports' / f'obsolete-paths-{revision}.txt'
            obsolete.write_text('\n'.join(stale) + '\n', encoding='utf-8')
            print(f'All required files verified, but {len(stale)} old remote paths remain. List: {obsolete}', flush=True)
            print('No old paths were deleted. Review this list before approving remote cleanup.', flush=True)
        else:
            print(f'Private dataset ready: https://huggingface.co/datasets/{repo_id}', flush=True)
        print(f'Dataset revision: {revision}')
        print('Download this revision and run the dataset audit to verify the transferred contents.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id', required=True, help='Hugging Face username-or-organization/BioCal3D-Data')
    parser.add_argument('--data-root', type=Path, help='Parent containing the four prepared release folders')
    parser.add_argument('--readme-only', action='store_true', help='Update the dataset README and verify that all other files are unchanged')
    parser.add_argument('--study-run-archive', type=Path, help='Publish the supplied final classification run.zip and its recorded manifest separately from the scans')
    parser.add_argument('--dry-run', action='store_true', help='Audit and report upload size without contacting Hugging Face')
    parser.add_argument('--audit-inventory', type=Path, help='Reuse a completed clean 512-scan audit of this unchanged release')
    args = parser.parse_args()
    if len(args.repo_id.split('/')) != 2 or not all(args.repo_id.split('/')):
        parser.error('Use a repository ID in the form username-or-organization/BioCal3D-Data.')
    if not args.readme_only and not args.study_run_archive and args.data_root is None:
        parser.error('--data-root is required for a data upload.')
    if args.readme_only and (args.dry_run or args.audit_inventory):
        parser.error('--readme-only cannot be combined with --dry-run or --audit-inventory.')
    if args.study_run_archive and (args.readme_only or args.dry_run or args.audit_inventory):
        parser.error('--study-run-archive cannot be combined with another upload mode or audit options.')
    folder = Path.home() / 'BioCal3D' / 'upload-logs'
    folder.mkdir(parents=True, exist_ok=True)
    log = folder / ('upload-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '.log')
    print(f'Upload log: {log}', flush=True)
    class Tee:
        def __init__(self, terminal, file): self.terminal, self.file = terminal, file
        def write(self, value):
            self.terminal.write(value)
            self.file.write(value)
            return len(value)
        def flush(self): self.terminal.flush(); self.file.flush()
        def isatty(self): return False
        def __getattr__(self, name): return getattr(self.terminal, name)
    with log.open('w', encoding='utf-8', buffering=1) as stream:
        with redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
            try:
                if args.study_run_archive:
                    from huggingface_hub import HfApi
                    from hub_transfer import upload_study_archive
                    manifest = Path(__file__).resolve().parents[1] / 'analysis' / 'classification-study' / 'archives' / 'final-run-2026-11-24.json'
                    upload_study_archive(HfApi(), args.repo_id, args.study_run_archive, manifest)
                elif args.readme_only:
                    from huggingface_hub import HfApi
                    from hub_transfer import update_readme
                    update_readme(HfApi(), args.repo_id, Path(__file__).resolve().parents[1] / 'docs' / 'dataset-card.md')
                else:
                    upload(args.repo_id, args.data_root, args.dry_run, audit_inventory=args.audit_inventory)
            except KeyboardInterrupt:
                print(f'Upload interrupted. Rerun to resume. Log: {log}', flush=True)
                raise SystemExit(130)
            except Exception:
                traceback.print_exc()
                print(f'Upload did not complete. Diagnostic log: {log}', flush=True)
                raise SystemExit(2)


if __name__ == '__main__':
    main()
