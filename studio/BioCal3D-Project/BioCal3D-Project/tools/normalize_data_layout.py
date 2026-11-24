"""Standardize scanner/group folders in a prepared release, with rollback backups."""
import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path, PureWindowsPath

from upload_dataset import validated_inventory, progress_report
from audit_datasets import VARIANTS
from prepare_data_release import ROI_METADATA, release_files


def plan_layout(root, rows):
    mapping = {}
    targets = {}
    for row in rows:
        group = re.fullmatch(r'BCG([1-6])T[012]-[1-8]', row['sample'])[1]
        for variant in VARIANTS:
            parts = row[variant].replace('\\', '/').split('/')
            offset = int(parts[0].casefold() == 'samples')
            if (len(parts) < offset + 3 or parts[offset].casefold() != row['scanner'].casefold()
                    or not re.fullmatch(r'G' + group + r'(?:[- _()].*)?', parts[offset + 1], re.I)):
                raise ValueError(f'Unrecognized scanner/group layout: {variant}/{row[variant]}')
            old = '/'.join((variant, *parts[:offset + 2]))
            new = f"{variant}/{row['scanner']}/G{group}"
            if new in targets and targets[new] != old:
                raise ValueError(f'Two source folders would merge into {new}; no changes made.')
            targets[new] = old
            mapping[old] = new
    for old, new in mapping.items():
        source, target = root / old, root / new
        if not source.is_dir() or any(p.is_symlink() for p in (source, *source.parents)):
            raise ValueError(f'Source is missing or uses a symbolic link: {source}')
        if old != new and target.exists():
            raise ValueError(f'Destination already exists: {target}')
    # Removing the samples wrapper must not conceal extra, unaudited contents.
    for variant in VARIANTS:
        wrapper = root / variant / 'samples'
        if not wrapper.is_dir():
            continue
        allowed = {root / old for old in mapping if old.startswith(variant + '/samples/')}
        for scanner in wrapper.iterdir():
            if not scanner.is_dir() or any(group not in allowed for group in scanner.iterdir()):
                raise ValueError(f'Unaudited contents in {scanner}; review before normalization.')
    return mapping


def mapped_string(value, mapping):
    if not isinstance(value, str):
        return value
    path = PureWindowsPath(value)
    if path.drive or path.root or '..' in path.parts:
        return value  # Preserve absolute provenance, not the current dataset location.
    normalized = value.replace('\\', '/')
    for old, new in sorted(mapping.items(), key=lambda item: -len(item[0])):
        if normalized == old or normalized.startswith(old + '/'):
            return new + normalized[len(old):]
    return value


def mapped_json(value, mapping):
    if isinstance(value, dict):
        return {key: mapped_json(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [mapped_json(item, mapping) for item in value]
    return mapped_string(value, mapping)


def csv_bytes(rows, columns):
    import io
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8-sig')


def normalize(root, inventory, *, apply=False, backup_parent=None):
    root = Path(root).expanduser().resolve()
    rows = validated_inventory(Path(inventory).expanduser().resolve())
    mapping = plan_layout(root, rows)
    changes = {old: new for old, new in mapping.items() if old != new}
    for old, new in changes.items():
        print(f'{old} -> {new}', flush=True)
    print(f'{len(changes)} group folders to relocate; model and ROI file bytes stay unchanged.', flush=True)
    if not apply:
        print('Preview only. Add --apply to perform this plan.', flush=True)
        return None
    if not changes:
        print('Layout is already standardized; nothing changed.', flush=True)
        return root / 'dataset_inventory.csv'
    updates = {}
    new_rows = []
    for row in rows:
        row = dict(row)
        for variant in VARIANTS:
            row[variant] = mapped_string(variant + '/' + row[variant].replace('\\', '/'), mapping)[len(variant) + 1:]
        new_rows.append(row)
    updates[root / 'dataset_inventory.csv'] = csv_bytes(new_rows, list(rows[0]))
    for variant in VARIANTS:
        local = {old[len(variant) + 1:]: new[len(variant) + 1:]
                 for old, new in mapping.items() if old.startswith(variant + '/')}
        for name in ROI_METADATA:
            path = root / variant / name
            if name == 'layout_manifest.json' or not path.is_file():
                continue
            if path.suffix == '.json':
                original = json.loads(path.read_text(encoding='utf-8-sig'))
                revised = mapped_json(original, local)
                if original != revised:
                    updates[path] = (json.dumps(revised, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
            elif path.suffix == '.csv':
                with path.open(encoding='utf-8-sig', newline='') as stream:
                    reader = csv.DictReader(stream)
                    columns = reader.fieldnames
                    original = list(reader)
                revised = [{key: mapped_string(value, local) for key, value in row.items()} for row in original]
                if original != revised:
                    updates[path] = csv_bytes(revised, columns)
        manifest = root / variant / 'layout_manifest.json'
        if manifest.exists():
            raise ValueError(f'A layout manifest already exists: {manifest}; review before a second migration.')
        updates[manifest] = (json.dumps({'schema': 1, 'folder_mapping': local,
                                      'note': 'Original group labels, including limitations, are retained in folder_mapping.'},
                                     indent=2) + '\n').encode('utf-8')
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    backup = Path(backup_parent or root.parent) / (root.name + '-layout-backup-' + stamp)
    backup.mkdir(parents=True, exist_ok=False)
    originals = {path: path.read_bytes() if path.exists() else None for path in updates}
    for path, content in originals.items():
        if content is not None:
            target = backup / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    (backup / 'completed_audit_before.csv').write_bytes(Path(inventory).read_bytes())
    (backup / 'rename_plan.json').write_text(json.dumps(changes, indent=2), encoding='utf-8')
    moved = []
    try:
        for number, (old, new) in enumerate(changes.items(), 1):
            target = root / new
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f'Relocating group {number}/{len(changes)}: {new}', flush=True)
            (root / old).rename(target)
            moved.append((old, new))
        for variant in VARIANTS:
            wrapper = root / variant / 'samples'
            if wrapper.is_dir():
                for scanner in wrapper.iterdir():
                    scanner.rmdir()
                wrapper.rmdir()
        for path, content in updates.items():
            path.write_bytes(content)
        for variant in VARIANTS:
            with progress_report(f'Verifying selected data and texture paths: {variant}') as report:
                for path in release_files(root / variant, new_rows, variant, root / VARIANTS[2], progress=report):
                    report(f'Checking relocated file: {path.relative_to(root)}')
                    if not path.is_file():
                        raise FileNotFoundError(path)
        print(f'Layout verified. Updated audit: {root / "dataset_inventory.csv"}', flush=True)
        print(f'Original labels, indexes and rename plan backed up: {backup}', flush=True)
    except BaseException:
        for old, new in reversed(moved):
            source = root / old
            source.parent.mkdir(parents=True, exist_ok=True)
            (root / new).rename(source)
        for path, content in originals.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)
        print(f'Operation failed; original folders and indexes restored. Backup: {backup}', flush=True)
        raise
    return root / 'dataset_inventory.csv'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--audit-inventory', type=Path, required=True,
                        help='Completed clean audit of this unchanged release')
    parser.add_argument('--apply', action='store_true', help='Apply the displayed folder normalization')
    args = parser.parse_args()
    normalize(args.data_root, args.audit_inventory, apply=args.apply)


if __name__ == '__main__':
    main()
