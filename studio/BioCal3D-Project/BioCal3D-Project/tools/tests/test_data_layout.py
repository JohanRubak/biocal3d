"""Real filesystem checks for migration, provenance, rollback and study readers."""
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import normalize_data_layout as layout
from audit_datasets import audit, VARIANTS, expected_keys
from dataset_paths import relocated_asset
from original_models import OriginalModelIndex
from roi_browser_core import discover
from upload_dataset import validated_inventory


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / 'Data'
        self.rows = []
        self.original_bytes = {}
        header = ('ply\nformat ascii 1.0\ncomment TextureFile shared.png\n'
                  'element vertex 3\nproperty float x\nproperty float y\nproperty float z\n'
                  'property uchar red\nproperty uchar green\nproperty uchar blue\n'
                  'element face 1\nproperty list uchar int vertex_indices\nend_header\n'
                  '0 0 0 255 0 0\n1 0 0 0 255 0\n0 1 0 0 0 255\n3 0 1 2\n')
        indices = {variant: [] for variant in VARIANTS[2:]}
        for scanner, sample in sorted(expected_keys()):
            group = sample[3]
            row = {'scanner': scanner, 'sample': sample, 'issues': ''}
            for variant in VARIANTS[:2]:
                group_name = f'G{group} (data with limitations)' if group == '5' else f'G{group}-OK'
                folder = self.root / variant / scanner / group_name
                if scanner == 'iTERO' and variant == VARIANTS[0]:
                    folder /= sample
                folder.mkdir(parents=True, exist_ok=True)
                mesh = folder / (sample + '.ply')
                mesh.write_text(header)
                (folder / 'shared.png').write_bytes(b'texture fixture')
                row[variant] = mesh.relative_to(self.root / variant).as_posix()
                row['raw_sha256' if variant == VARIANTS[0] else 'harmonized_sha256'] = digest(mesh)
            for variant in VARIANTS[2:]:
                folder = self.root / variant / 'samples' / scanner / f'G{group}' / sample
                folder.mkdir(parents=True)
                mesh = folder / 'geometry.npz'
                native = row[VARIANTS[0]]
                np.savez(mesh, vertices=np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]]),
                         faces=np.array([[0, 1, 2]]), original_to_canonical=np.eye(4),
                         source_ply=f'C:/old/raw_native/{native}', source_sha256=row['raw_sha256'],
                         harmonized_source_ply=f'C:/old/full_harmonized/{row[VARIANTS[1]]}',
                         harmonized_source_sha256=row['harmonized_sha256'],
                         texture_path=f'C:/old/raw_native/{Path(native).parent}/shared.png')
                row[variant] = mesh.relative_to(self.root / variant).as_posix()
                indices[variant].append({'sample_dir': folder.relative_to(self.root / variant).as_posix(),
                                         'geometry_sha256': digest(mesh), 'scanner': scanner,
                                         'sample_id': sample, 'status': 'OK'})
            self.rows.append(row)
        for variant, records in indices.items():
            (self.root / variant / 'dataset_index.json').write_text(json.dumps({'scans': records}))
            (self.root / variant / 'scan_manifest.csv').write_bytes(layout.csv_bytes(records, list(records[0])))
        self.inventory = self.base / 'audit.csv'
        self.inventory.write_bytes(layout.csv_bytes(self.rows, list(self.rows[0])))
        self.original_bytes = {p.relative_to(self.root).as_posix(): digest(p)
                               for p in self.root.rglob('*') if p.suffix in ('.npz', '.ply', '.png')}

    def tearDown(self):
        self.temp.cleanup()

    def run_migration(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return layout.normalize(self.root, self.inventory, **kwargs)

    def test_preview_changes_nothing(self):
        self.assertIsNone(self.run_migration())
        self.assertTrue((self.root / VARIANTS[2] / 'samples').is_dir())
        self.assertFalse(list(self.base.glob('Data-layout-backup-*')))

    def test_normalization_preserves_bytes_and_all_512_scan_correspondences(self):
        result = self.run_migration(apply=True)
        rows = validated_inventory(result)
        for old, sha in self.original_bytes.items():
            for variant in VARIANTS:
                if old.startswith(variant + '/'):
                    from dataset_paths import canonical_relative
                    new = variant + '/' + canonical_relative(old[len(variant) + 1:])
                    self.assertEqual(digest(self.root / new), sha)
        for variant in VARIANTS[2:]:
            self.assertFalse((self.root / variant / 'samples').exists())
            self.assertEqual(len(discover(self.root / variant)[1]), 512)
            index = json.loads((self.root / variant / 'dataset_index.json').read_text())
            for record in index['scans']:
                self.assertTrue((self.root / variant / record['sample_dir'] / 'geometry.npz').is_file())
                self.assertFalse(record['sample_dir'].startswith('samples/'))
        test_row = next(row for row in rows if row['scanner'] == 'TRIOS5' and row['sample'] == 'BCG5T0-1')
        old = 'C:/old/raw_native/TRIOS5/G5 (data with limitations)/shared.png'
        texture = relocated_asset(self.root / VARIANTS[2], old)
        self.assertEqual(texture, self.root / VARIANTS[0] / 'TRIOS5/G5/shared.png')
        index = OriginalModelIndex(self.root / VARIANTS[1])
        with patch.object(index, '_scan', side_effect=AssertionError('legacy hints must resolve directly')):
            found = index.resolve(['C:/old/full_harmonized/TRIOS5/G5 (data with limitations)/BCG5T0-1.ply'],
                                  scanner='TRIOS5', sample='BCG5T0-1', checksum=test_row['harmonized_sha256'])
        self.assertEqual(found, self.root / VARIANTS[1] / test_row[VARIANTS[1]])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(audit([self.root / v for v in VARIANTS], self.base / 'after.csv'), 0)
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'analysis' / 'classification-study'))
        from study_data import load_snapshot
        for variant in VARIANTS[2:]:
            records, snapshot_audit, excluded = load_snapshot(self.root / variant, approved_export=True)
            self.assertEqual(len(records), 512)
            self.assertEqual(snapshot_audit['physical_specimens'], 48)
            self.assertFalse(excluded)
        manifest = json.loads((self.root / VARIANTS[1] / 'layout_manifest.json').read_text())
        self.assertTrue(any('data with limitations' in key for key in manifest['folder_mapping']))
        self.assertTrue(list(self.base.glob('Data-layout-backup-*/rename_plan.json')))
        # A second normalization is a no-op.
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(layout.normalize(self.root, result, apply=True), result)

    def test_failed_verification_restores_original_paths_and_metadata(self):
        before = (self.root / VARIANTS[2] / 'dataset_index.json').read_bytes()
        with patch.object(layout, 'release_files', side_effect=FileNotFoundError('missing texture')):
            with self.assertRaises(FileNotFoundError):
                self.run_migration(apply=True)
        for old, sha in self.original_bytes.items():
            self.assertEqual(digest(self.root / old), sha)
        self.assertEqual((self.root / VARIANTS[2] / 'dataset_index.json').read_bytes(), before)
        self.assertFalse((self.root / 'dataset_inventory.csv').exists())
        self.assertFalse((self.root / VARIANTS[2] / 'layout_manifest.json').exists())

    def test_collision_is_rejected_without_merging_groups(self):
        (self.root / VARIANTS[1] / 'TRIOS5/G5').mkdir()
        with self.assertRaisesRegex(ValueError, 'Destination already exists'):
            self.run_migration(apply=True)
        self.assertFalse(list(self.base.glob('Data-layout-backup-*')))

    def test_full_model_renaming_preserves_hashes_stem_textures_and_old_provenance(self):
        from rename_full_models import rename_models
        from prepare_data_release import release_files
        from dataset_paths import renamed_texture
        from mesh_io import find_texture
        from roi_browser_core import find_texture as browser_texture
        original = self.rows[0]
        variant = VARIANTS[0]
        source = self.root / variant / original[variant]
        old = source.with_name('scanner_export.ply')
        # Make texture discovery depend on the historical mesh stem.
        old.write_bytes(source.read_bytes().replace(b'comment TextureFile shared.png\n', b''))
        source.unlink()
        sidecar = old.with_name('scanner_export_texture.png')
        sidecar.write_bytes(b'original texture pixels')
        original[variant] = old.relative_to(self.root / variant).as_posix()
        original['raw_sha256'] = digest(old)
        self.inventory.write_bytes(layout.csv_bytes(self.rows, list(self.rows[0])))
        old_sha, image_sha = digest(old), digest(sidecar)
        with contextlib.redirect_stdout(io.StringIO()):
            updated = rename_models(self.root, self.inventory, apply=True)
        row = validated_inventory(updated)[0]
        canonical = self.root / variant / row[variant]
        self.assertEqual(canonical.name, row['sample'] + '.ply')
        self.assertEqual(digest(canonical), old_sha)
        self.assertEqual(digest(sidecar), image_sha)
        self.assertIn(sidecar, release_files(self.root / variant, [row], variant, self.root / VARIANTS[2]))
        self.assertEqual(renamed_texture(canonical), sidecar)
        self.assertEqual(find_texture(canonical), sidecar)
        self.assertEqual(browser_texture('', canonical, self.root / variant), sidecar)
        index = OriginalModelIndex(self.root / variant)
        with patch.object(index, '_scan', side_effect=AssertionError('no tree walk expected')):
            self.assertEqual(index.resolve([str(old)], scanner=row['scanner'], sample=row['sample'],
                                           checksum=row['raw_sha256']), canonical)
        # A second run preserves the old-name manifest.
        manifest = (self.root / variant / 'layout_manifest.json').read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(rename_models(self.root, updated, apply=True), updated)
        self.assertEqual((self.root / variant / 'layout_manifest.json').read_bytes(), manifest)

    def test_filename_collision_and_failed_verification_never_lose_the_original(self):
        import rename_full_models as names
        row = self.rows[0]
        variant = VARIANTS[0]
        canonical = self.root / variant / row[variant]
        old = canonical.with_name('scanner_' + canonical.name)
        canonical.rename(old)
        row[variant] = old.relative_to(self.root / variant).as_posix()
        self.inventory.write_bytes(layout.csv_bytes(self.rows, list(self.rows[0])))
        before = self.inventory.read_bytes()
        canonical.write_bytes(b'different mesh already present')
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            with contextlib.redirect_stdout(io.StringIO()):
                names.rename_models(self.root, self.inventory, apply=True)
        self.assertTrue(old.exists())
        canonical.unlink()
        with patch.object(names, 'release_files', side_effect=OSError('simulated verification error')):
            with self.assertRaisesRegex(OSError, 'simulated'):
                with contextlib.redirect_stdout(io.StringIO()):
                    names.rename_models(self.root, self.inventory, apply=True)
        self.assertTrue(old.is_file())
        self.assertFalse(canonical.exists())
        self.assertEqual(self.inventory.read_bytes(), before)
        self.assertFalse((self.root / variant / 'layout_manifest.json').exists())

    def test_release_builder_normalizes_copy_and_preserves_working_source(self):
        from prepare_data_release import prepare
        destination = self.base / 'published'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(prepare(self.root, destination), destination)
        self.assertEqual(len(validated_inventory(destination / 'dataset_inventory.csv')), 512)
        self.assertFalse((destination / VARIANTS[2] / 'samples').exists())
        self.assertTrue((self.root / VARIANTS[2] / 'samples').is_dir())
        self.assertTrue((destination / VARIANTS[0] / 'layout_manifest.json').is_file())


if __name__ == '__main__':
    unittest.main()
