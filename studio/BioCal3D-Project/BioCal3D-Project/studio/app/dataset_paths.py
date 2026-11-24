"""Folder names and defaults shared by Studio and the research commands."""
import os
import re
import json
from pathlib import Path

RAW_NATIVE = 'raw_native'
FULL_HARMONIZED = 'full_harmonized'
APPROVED_REORIENTED = 'approved_rois_reoriented'
APPROVED_HARMONIZED = 'approved_rois_reoriented_harmonized'
INITIAL_ROIS = 'initial_rois'
RELEASE_FOLDERS = (RAW_NATIVE, FULL_HARMONIZED, APPROVED_REORIENTED, APPROVED_HARMONIZED)
LEGACY_FOLDERS = {
    '02-09-2026-data collected': RAW_NATIVE,
    'color-harmonized data': FULL_HARMONIZED,
    'approved rois_reoriented': APPROVED_REORIENTED,
    'approved rois_reoriented_harmonized': APPROVED_HARMONIZED,
}


def canonical_relative(recorded):
    """Translate legacy scanner/group paths without changing acquisition basenames."""
    parts = str(recorded).replace('\\', '/').split('/')
    if parts and parts[0].casefold() == 'samples':
        parts = parts[1:]
    scanners = {name.casefold(): name for name in ('iTERO', 'LABscanner', 'TRIOS3', 'TRIOS5')}
    if len(parts) >= 2 and parts[0].casefold() in scanners:
        parts[0] = scanners[parts[0].casefold()]
        group = re.fullmatch(r'G([1-6])(?:[- _()].*)?', parts[1], re.I)
        if group:
            parts[1] = 'G' + group[1]
    return '/'.join(parts)


def data_root():
    configured = os.environ.get('BIOCAL3D_DATA_ROOT')
    return Path(configured).expanduser().resolve() if configured else Path(__file__).resolve().parents[2] / 'data'


def renamed_texture(mesh):
    """Find an unmodified scanner texture sidecar after a mesh-only rename."""
    mesh = Path(mesh)
    root = next((parent for parent in mesh.parents if parent.name.casefold() in RELEASE_FOLDERS), None)
    if root is None:
        return None
    manifest = root / 'layout_manifest.json'
    if not manifest.is_file():
        return None
    mapping = json.loads(manifest.read_text(encoding='utf-8-sig')).get('file_mapping', {})
    relative = mesh.relative_to(root).as_posix()
    stems = set()
    for old, new in mapping.items():
        if new == relative:
            stem = Path(old).stem.casefold()
            stems.update({stem, stem + '_texture', stem + '.texture'})
    matches = [path for path in mesh.parent.iterdir() if path.stem.casefold() in stems
               and path.suffix.casefold() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'} and path.is_file()]
    return matches[0] if len(matches) == 1 else None


def full_model_sibling(roi_root):
    folder = {APPROVED_REORIENTED: RAW_NATIVE, APPROVED_HARMONIZED: FULL_HARMONIZED}.get(roi_root.name.casefold())
    sibling = roi_root.parent / folder if folder else None
    return sibling if sibling and sibling.is_dir() else None


def relocated_asset(dataset, recorded):
    """Resolve a saved Windows or POSIX asset path under a relocated release."""
    parts = recorded.replace('\\', '/').split('/')
    release = next((parent.parent for parent in (dataset, *dataset.parents)
                    if parent.name.casefold() in RELEASE_FOLDERS), None)
    if release is None or '..' in parts:
        return None
    for index, part in enumerate(parts[:-1]):
        name = part.casefold()
        folder = LEGACY_FOLDERS.get(name, name)
        if folder in RELEASE_FOLDERS:
            relative = '/'.join(parts[index + 1:])
            for value in (relative, canonical_relative(relative)):
                candidate = release / folder / value
                if candidate.resolve().is_relative_to(release.resolve()) and candidate.is_file():
                    return candidate
    return None
