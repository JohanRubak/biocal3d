"""Rename audited full-model PLY files to BCG<group>T<timepoint>-<specimen>.ply."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from audit_datasets import VARIANTS
from normalize_data_layout import csv_bytes, mapped_json, mapped_string
from prepare_data_release import ROI_METADATA, release_files
from upload_dataset import validated_inventory, progress_report


def rename_models(root, inventory, *, apply=False, backup_parent=None):
    root = Path(root).expanduser().resolve()
    inventory = Path(inventory).expanduser().resolve()
    rows = validated_inventory(inventory)
    changes = {}
    for row in rows:
        for variant in VARIANTS[:2]:
            old = Path(variant) / row[variant].replace('\\', '/')
            if old.suffix.lower() != '.ply':
                continue
            new = old.with_name(row['sample'] + '.ply')
            if old.as_posix() == new.as_posix():
                continue
            source, target = root / old, root / new
            if not source.is_file() or source.is_symlink():
                raise ValueError(f'Expected ordinary audited mesh file: {source}')
            if target.exists() and target != source:
                raise ValueError(f'Renaming would overwrite {target}; no files changed.')
            if new.as_posix() in changes.values():
                raise ValueError(f'Two meshes would have the same target: {new}')
            changes[old.as_posix()] = new.as_posix()
    print(f'{len(changes)} full-model PLY files need scan-ID filenames.', flush=True)
    for old, new in changes.items():
        print(f'{old} -> {new}', flush=True)
    if not apply:
        print('Preview only. Add --apply to perform the renames.', flush=True)
        return None
    if not changes:
        print('Full-model filenames are already consistent.', flush=True)
        return inventory
    revised = []
    for row in rows:
        row = dict(row)
        for variant in VARIANTS[:2]:
            name = variant + '/' + row[variant].replace('\\', '/')
            row[variant] = changes.get(name, name)[len(variant) + 1:]
        revised.append(row)
    updates = {root / 'dataset_inventory.csv': csv_bytes(revised, list(rows[0]))}
    for variant in VARIANTS[:2]:
        prefix = variant + '/'
        local = {old[len(prefix):]: new[len(prefix):] for old, new in changes.items() if old.startswith(prefix)}
        if not local:
            continue
        manifest_path = root / variant / 'layout_manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig')) if manifest_path.exists() else {'schema': 1}
        previous = manifest.get('file_mapping', {})
        manifest['file_mapping'] = {old: local.get(new, new) for old, new in previous.items()}
        manifest['file_mapping'].update(local)
        manifest['filename_note'] = 'PLY names use the canonical scan ID; original names and unchanged byte checksums are retained.'
        updates[manifest_path] = (json.dumps(manifest, indent=2) + '\n').encode('utf-8')
        for name in ROI_METADATA:
            path = root / variant / name
            if not path.is_file() or name == 'layout_manifest.json':
                continue
            if path.suffix == '.json':
                before = json.loads(path.read_text(encoding='utf-8-sig'))
                after = mapped_json(before, local)
                if before != after:
                    updates[path] = (json.dumps(after, indent=2) + '\n').encode('utf-8')
            elif path.suffix == '.csv':
                with path.open(encoding='utf-8-sig', newline='') as stream:
                    reader = csv.DictReader(stream); columns = reader.fieldnames; before = list(reader)
                after = [{key: mapped_string(value, local) for key, value in record.items()} for record in before]
                if before != after:
                    updates[path] = csv_bytes(after, columns)
    backup = Path(backup_parent or root.parent) / (root.name + '-filenames-backup-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    backup.mkdir(parents=True, exist_ok=False)
    original = {path: path.read_bytes() if path.exists() else None for path in updates}
    for path, content in original.items():
        if content is not None:
            destination = backup / path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
    (backup / 'completed_audit_before.csv').write_bytes(inventory.read_bytes())
    (backup / 'rename_plan.json').write_text(json.dumps(changes, indent=2), encoding='utf-8')
    moved = []
    try:
        for number, (old, new) in enumerate(changes.items(), 1):
            print(f'Renaming PLY {number}/{len(changes)}: {new}', flush=True)
            (root / old).rename(root / new)
            moved.append((old, new))
        for path, content in updates.items():
            path.write_bytes(content)
        for variant in VARIANTS:
            with progress_report(f'Verifying renamed data and textures: {variant}') as report:
                paths = release_files(root / variant, revised, variant, root / VARIANTS[2], progress=report)
                for path in paths:
                    report(f'Checking {path.relative_to(root)}')
                    if not path.is_file():
                        raise FileNotFoundError(path)
    except BaseException:
        for old, new in reversed(moved):
            (root / new).rename(root / old)
        for path, content in original.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)
        print(f'Original names and indexes restored. Backup: {backup}', flush=True)
        raise
    print(f'Filename verification passed. Updated inventory: {root / "dataset_inventory.csv"}', flush=True)
    print(f'Original names/indexes backed up: {backup}', flush=True)
    print('Mesh and texture bytes are unchanged. ROI packages retain roi.ply and geometry.npz.', flush=True)
    return root / 'dataset_inventory.csv'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--audit-inventory', required=True, type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    rename_models(args.data_root, args.audit_inventory, apply=args.apply)


if __name__ == '__main__':
    main()
