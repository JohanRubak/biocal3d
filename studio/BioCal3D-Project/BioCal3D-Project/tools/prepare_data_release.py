"""Copy the four BioCal3D datasets into a validated handover release."""
import argparse
import csv
import json
import os
import re
from pathlib import Path, PureWindowsPath
import shutil
import tempfile

from audit_datasets import VARIANTS, audit
from browser_original_models import TEXTURE_KEYS
from dataset_paths import LEGACY_FOLDERS, canonical_relative


ROI_FILES = (
    'geometry.npz', 'roi.ply', 'roi.vtp', 'maps.npz', 'metadata.json', 'meta.json',
    'rgb.png', 'valid_mask.png',
)
ROI_METADATA = (
    'dataset_index.json', 'scan_manifest.csv', 'annotation_inventory.csv',
    'scanner_orientation.json', 'geometry_rebuild.json', 'orientation_corrections_used.json',
    'run_config.json', 'harmonized_roi_audit.csv', 'splits.json', 'split_assignments.csv',
    'specimen_splits.csv', 'splits.csv', 'classes.txt', 'class_map.json', 'annotation_classes.json',
    'layout_manifest.json',
)
IMAGE_TYPES = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def source_folders(source):
    accepted = {**LEGACY_FOLDERS, **{name: name for name in VARIANTS}}
    matches = {name: [] for name in VARIANTS}
    for folder in source.iterdir():
        variant = accepted.get(folder.name.casefold())
        if variant and folder.is_dir():
            matches[variant].append(folder.resolve())
    for variant, candidates in matches.items():
        if len(candidates) != 1:
            names = ', '.join(str(path) for path in candidates) or 'none'
            raise ValueError(f'Expected one source folder for {variant}. Found {names}.')
    roots = [matches[name][0] for name in VARIANTS]
    if len(set(roots)) != len(roots):
        raise ValueError('The four source folders must resolve to distinct directories.')
    return roots


def directory_files(folder, cache):
    if folder not in cache:
        with os.scandir(folder) as entries:
            cache[folder] = {Path(entry.path) for entry in entries if entry.is_file()}
    return cache[folder]


def texture_hints(mesh, cache):
    """Read only texture fields, using one directory listing for existence checks."""
    import numpy as np
    files = directory_files(mesh.parent, cache)
    result = {}
    geometry = mesh if mesh.suffix.casefold() == '.npz' else mesh.with_name('geometry.npz')
    if geometry in files:
        with np.load(geometry, allow_pickle=False) as arrays:
            for key in TEXTURE_KEYS:
                if key in arrays.files:
                    value = arrays[key]
                    if value.size == 1:
                        result[key] = str(value.item())

    def fields(item):
        if isinstance(item, dict):
            for key, value in item.items():
                if key in TEXTURE_KEYS and key not in result:
                    result[key] = value
                if isinstance(value, dict):
                    fields(value)
    for name in ('meta.json', 'metadata.json', 'source.json', 'review.json', 'sample.json'):
        metadata = mesh.with_name(name)
        if metadata in files:
            try:
                data = json.loads(metadata.read_text(encoding='utf-8-sig'))
            except (ValueError, OSError):
                continue
            fields(data)
    return result


def resolve_texture(root, mesh, hint, cache=None):
    cache = {} if cache is None else cache
    recorded = PureWindowsPath(str(hint).strip().strip('"'))
    if not recorded.name:
        return None
    candidates = [mesh.parent / recorded.name, Path(str(hint))]
    if not recorded.anchor:
        candidates.insert(0, mesh.parent.joinpath(*recorded.parts))
    for index, part in enumerate(recorded.parts[:-1]):
        if part.casefold() == root.name.casefold():
            candidates.append(root.joinpath(*recorded.parts[index + 1:]))
            candidates.append(root / canonical_relative('/'.join(recorded.parts[index + 1:])))
        if part.casefold() in {'itero', 'labscanner', 'trios3', 'trios5'}:
            candidates.append(root / canonical_relative('/'.join(recorded.parts[index:])))
    for candidate in dict.fromkeys(candidates):
        if candidate.suffix.casefold() not in IMAGE_TYPES:
            continue
        # Ignore saved absolute paths outside the selected dataset before IO.
        if not candidate.is_relative_to(root):
            continue
        try:
            exists = candidate in directory_files(candidate.parent, cache)
        except FileNotFoundError:
            continue
        if exists:
            resolved = candidate.resolve()
            if resolved.is_relative_to(root):
                return resolved
    return None



def scan_textures(mesh, root, cache=None):
    """Keep PLY-declared textures and scanner sidecars named for the mesh."""
    cache = {} if cache is None else cache
    textures = set()
    if mesh.suffix.casefold() == '.ply':
        with mesh.open('rb') as stream:
            for _ in range(4096):
                line = stream.readline(8192)
                if not line:
                    raise ValueError(f"PLY header is incomplete: {mesh}")
                text = line.decode('ascii', errors='replace').strip()
                if text == 'end_header':
                    break
                hint = re.match(r'^comment\s+TextureFile\s+(.+)$', text, re.I)
                if hint:
                    texture = resolve_texture(root, mesh, hint[1], cache)
                    if texture is None:
                        raise FileNotFoundError(f'Texture declared by {mesh} is unavailable: {hint[1]}')
                    textures.add(texture)
            else:
                raise ValueError(f"PLY header is too long: {mesh}")
    stems = {mesh.stem.casefold(), f'{mesh.stem.casefold()}_texture', f'{mesh.stem.casefold()}.texture'}
    manifest_key = ('file_mapping', root)
    if manifest_key not in cache:
        manifest = root / 'layout_manifest.json'
        cache[manifest_key] = json.loads(manifest.read_text(encoding='utf-8-sig')).get('file_mapping', {}) if manifest in directory_files(root, cache) else {}
    relative = mesh.relative_to(root).as_posix()
    for old, new in cache[manifest_key].items():
        if new == relative:
            stem = Path(old).stem.casefold()
            stems.update({stem, stem + '_texture', stem + '.texture'})
    if mesh.name in {'roi.vtp', 'roi.ply', 'geometry.npz'}:
        stems.update({'texture', 'roi.texture', 'roi_texture'})
    for candidate in directory_files(mesh.parent, cache):
        if candidate.suffix.casefold() in IMAGE_TYPES and candidate.stem.casefold() in stems:
            textures.add(candidate)
    return textures


def release_files(root, rows, variant, approved_root, *, progress=None):
    files = set()
    is_roi = variant in VARIANTS[2:]
    cache = {}
    report = progress or (lambda message: None)
    for number, row in enumerate(rows, 1):
        status = f"Selecting {variant}: scan {number}/{len(rows)} — {row['scanner']}/{row['sample']}"
        report(status + " — reading mesh header and texture references")
        mesh = root / row[variant]
        files.add(mesh)
        files.update(scan_textures(mesh, root, cache))
        if variant == VARIANTS[0] or is_roi:
            metadata_path = mesh if is_roi else approved_root / row[VARIANTS[2]]
            report(status + " — reading texture metadata")
            metadata = texture_hints(metadata_path, cache)
            for key in TEXTURE_KEYS:
                if metadata.get(key):
                    texture = resolve_texture(root, mesh, metadata[key], cache)
                    if texture is not None:
                        files.add(texture)
        if is_roi:
            for name in ROI_FILES:
                sibling = mesh.with_name(name)
                if sibling in directory_files(mesh.parent, cache):
                    files.add(sibling)
                    if sibling.suffix in {'.ply', '.vtp'}:
                        files.update(scan_textures(sibling, root, cache))
    if is_roi:
        files.update(root / name for name in ROI_METADATA if root / name in directory_files(root, cache))
    manifest = root / 'layout_manifest.json'
    if manifest in directory_files(root, cache):
        files.add(manifest)
    return sorted(files)


def publish(release, destination):
    if not destination.exists():
        release.rename(destination)
        return
    names = (*VARIANTS, 'dataset_inventory.csv')
    with (destination / '.biocal3d-release.lock').open('x') as lock:
        try:
            conflicts = [name for name in names if (destination / name).exists()]
            if conflicts:
                raise FileExistsError(f'Release items already exist under {destination}: {", ".join(conflicts)}')
            moved = []
            try:
                for name in names:
                    (release / name).rename(destination / name)
                    moved.append(name)
            except OSError:
                for name in reversed(moved):
                    (destination / name).rename(release / name)
                raise
        finally:
            lock.close()
            (destination / '.biocal3d-release.lock').unlink()


def prepare(source, destination):
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    roots = source_folders(source)
    if destination.is_relative_to(source) or any(
        destination.is_relative_to(root) or root.is_relative_to(destination) for root in roots
    ):
        raise ValueError('Choose a release destination outside the source dataset tree.')
    if destination.exists() and not destination.is_dir():
        raise ValueError(f'Release destination is not a folder: {destination}')
    conflicts = [name for name in (*VARIANTS, 'dataset_inventory.csv') if (destination / name).exists()]
    if conflicts:
        raise FileExistsError(f'Release items already exist under {destination}: {", ".join(conflicts)}')

    staging_parent = destination if destination.is_dir() else destination.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.biocal3d-release-', dir=staging_parent) as staging:
        stage = Path(staging)
        source_inventory = stage / 'source_inventory.csv'
        if audit(roots, source_inventory):
            failure_report = stage.with_suffix('.source-audit.csv')
            shutil.copy2(source_inventory, failure_report)
            raise ValueError(f'Source verification failed. See {failure_report}.')
        with source_inventory.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        release = stage / 'release'
        copies = []
        for variant, original in zip(VARIANTS, roots):
            copy = release / variant
            copies.append(copy)
            files = release_files(original, rows, variant, roots[2])
            print(f'Copying {len(files):,} data files -> {variant}', flush=True)
            for path in files:
                if not path.resolve().is_relative_to(original):
                    raise ValueError(f'Data file points outside its source folder: {path}')
                target = copy / path.relative_to(original)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        inventory = release / '.verified_inventory.csv'
        # Standardize the copied release, leaving the working source untouched.
        from normalize_data_layout import normalize
        normalize(release, source_inventory, apply=True, backup_parent=stage)
        from rename_full_models import rename_models
        rename_models(release, release / 'dataset_inventory.csv' if (release / 'dataset_inventory.csv').exists()
                      else source_inventory, apply=True, backup_parent=stage)
        if audit(copies, inventory):
            failure_report = stage.with_suffix('.audit.csv')
            shutil.copy2(inventory, failure_report)
            raise ValueError(f'Release verification failed. See {failure_report}.')
        inventory.replace(release / 'dataset_inventory.csv')
        publish(release, destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Parent of the four working dataset folders')
    parser.add_argument('--destination', type=Path, required=True, help='Shared data folder, existing or new')
    args = parser.parse_args()
    try:
        release = prepare(args.source, args.destination)
    except (OSError, ValueError) as error:
        parser.exit(2, f'{error}\n')
    print(f'Release ready: {release}')
    print(f'Inventory: {release / "dataset_inventory.csv"}')


if __name__ == '__main__':
    main()
