"""Copy saved scanner reference fits into a working initial-ROI run."""
import hashlib
import json
from pathlib import Path
import shutil


def seed_reference(scanner, output_root, source_root=None, *, reference_sample_id=None):
    destination = Path(output_root) / scanner / 'reference_roi.npz'
    if destination.exists() or source_root is None:
        return destination
    source_root = Path(source_root).expanduser().resolve()
    source = source_root / scanner / 'reference_roi.npz'
    if not source.is_file():
        raise FileNotFoundError(f'Saved reference fit is missing: {source}')
    manifest = source_root / 'manifest.json'
    if manifest.is_file():
        entries = json.loads(manifest.read_text(encoding='utf-8'))['references']
        entry = next((item for item in entries if item['scanner'] == scanner), None)
        if entry is None or hashlib.sha256(source.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f'Saved reference checksum does not match its manifest: {scanner}')
    import numpy as np
    with np.load(source, allow_pickle=False) as data:
        required = {'scanner', 'center', 'normal', 'u_axis', 'v_axis', 'radius', 'roi_radius', 'z_offset', 'reference_sample_id'}
        if not required.issubset(data.files) or str(data['scanner'].item()) != scanner:
            raise ValueError(f'Saved reference has the wrong scanner/schema: {source}')
        if reference_sample_id and str(data['reference_sample_id'].item()) != reference_sample_id:
            raise ValueError(f'Saved reference sample does not match the configured reference scan: {scanner}')
        for key in ('center', 'normal', 'u_axis', 'v_axis'):
            if data[key].shape != (3,) or not np.isfinite(data[key]).all():
                raise ValueError(f'Invalid {key} in {source}')
        for key in ('radius', 'roi_radius'):
            if not np.isfinite(data[key]).all() or float(data[key]) <= 0:
                raise ValueError(f'Invalid {key} in {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation keeps an existing working fit safe from replacement.
    with destination.open('xb') as stream:
        stream.write(source.read_bytes())
    shutil.copystat(source, destination)
    print(f'Reused saved {scanner} reference fit: {destination}', flush=True)
    return destination
