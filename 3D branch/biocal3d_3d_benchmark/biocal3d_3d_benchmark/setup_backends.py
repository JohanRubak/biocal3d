#!/usr/bin/env python3
"""Fetch official model repositories; never changes the Python environment.

python setup_backends.py
Install GPU dependencies separately using README.md, then run --preflight.
Existing repositories are retained at their current commits; no automatic pull.
"""
import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backends-root', type=Path, default=Path(__file__).resolve().parent / 'third_party')
    args = p.parse_args()
    roots = [
        ('TSGCNet', 'https://github.com/ZhangLingMing1/TSGCNet.git'),
        ('PointNeXt', 'https://github.com/guochengqian/PointNeXt.git'),
        ('sonata', 'https://github.com/facebookresearch/sonata.git'),
        ('concerto', 'https://github.com/Pointcept/Concerto.git')]
    args.backends_root.mkdir(parents=True, exist_ok=True)
    versions = {}
    for name, url in roots:
        dest = args.backends_root / name
        if not dest.exists():
            subprocess.run(['git', 'clone', url, str(dest)], check=True)
        else:
            print(f'Reusing {dest}; not updating its commit.')
        versions[name] = subprocess.check_output(['git', '-C', str(dest), 'rev-parse', 'HEAD'], text=True).strip()
    # The upstream submodule URL uses SSH. Use the author's HTTPS repository
    # directly, avoiding an unnecessary GitHub SSH-account requirement.
    openpoints = args.backends_root / 'PointNeXt' / 'openpoints'
    if not (openpoints / 'models').is_dir():
        if openpoints.exists() and not any(openpoints.iterdir()):
            openpoints.rmdir()
        subprocess.run(['git', 'clone', 'https://github.com/guochengqian/openpoints.git', str(openpoints)], check=True)
    versions['openpoints'] = subprocess.check_output(['git', '-C', str(openpoints), 'rev-parse', 'HEAD'], text=True).strip()
    (args.backends_root / 'backend_versions.json').write_text(json.dumps(versions, indent=2))
    print('Official sources fetched. Install CUDA dependencies using README.md.')
    print('Next: python biocal3d_train_3d_models.py --data-root "YOUR_DATA_ROOT" --preflight')


if __name__ == '__main__':
    main()
