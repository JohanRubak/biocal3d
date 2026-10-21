from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import roi_browser_core as core
from original_models import OriginalModels
from test_core import sample


def original(folder, name='full.ply', textured=False):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    header = ['ply', 'format ascii 1.0']
    if textured: header.append('comment TextureFile full_texture.png')
    header += ['element vertex 4', 'property float x', 'property float y', 'property float z']
    if not textured: header += ['property uchar red', 'property uchar green', 'property uchar blue']
    header += ['element face 2', 'property list uchar int vertex_indices']
    if textured: header += ['property list uchar float texcoord']
    header.append('end_header')
    rows = ['98 -2 0', '103 -2 0', '98 3 0', '103 3 0']
    if not textured: rows = [r + ' 210 45 80' for r in rows]
    rows += ['3 0 1 2' + (' 6 0 0 1 0 0 1' if textured else ''),
             '3 1 3 2' + (' 6 1 0 1 1 0 1' if textured else '')]
    path.write_text('\n'.join(header + rows) + '\n')
    if textured:
        from PIL import Image
        Image.new('RGB', (32, 32), (160, 60, 100)).save(folder / 'full_texture.png')
    return path


class OriginalModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.base = Path(self.temp.name)
        self.root = self.base / 'data' / 'Approved ROIs'
        self.raw = self.base / 'data' / '02-09-2026-Data collected'
        self.path = sample(self.root)
        self.scan = core.discover(self.root)[1][0]
    def tearDown(self): self.temp.cleanup()
    def metadata(self, **extra):
        with np.load(self.path) as data: arrays = {k: data[k] for k in data.files}
        arrays.update(extra); np.savez(self.path, **arrays)

    def test_rebase_windows_path_and_apply_saved_transform_without_editing_inputs(self):
        source = original(self.raw / 'iTERO' / 'G3-OK' / 'BCG3T0-1')
        transform = np.eye(4); transform[0, 3] = -100
        self.metadata(source_ply=r'C:\Users\old\data\02-09-2026-Data collected\iTERO\G3-OK\BCG3T0-1\full.ply', original_to_canonical=transform)
        before = source.read_bytes(), self.path.read_bytes()
        resolver = OriginalModels(self.root, self.raw)
        mesh = resolver.load(self.scan)
        self.assertEqual(mesh['loaded_path'], str(source.resolve()))
        np.testing.assert_allclose(mesh['bounds'][:4], [-2.5, 2.5, -2.5, 2.5])
        self.assertEqual(mesh['model_representation'], 'original')
        self.assertIn('aligned', mesh['coordinate_frame'])
        self.assertEqual(before, (source.read_bytes(), self.path.read_bytes()))

    def test_matching_scanner_and_timepoint_with_ply_preferred_over_stl(self):
        source = original(self.raw / 'iTERO' / 'G3-OK' / 'BCG3T0-1')
        (source.parent / 'full.stl').touch()
        original(self.raw / 'TRIOS3' / 'G3' / 'BCG3T0-1')
        original(self.raw / 'iTERO' / 'G3' / 'BCG3T1-1')
        mesh = OriginalModels(self.root, self.raw).load(self.scan)
        self.assertEqual(mesh['loaded_path'], str(source.resolve()))
        self.assertIn('not aligned', mesh['coordinate_frame'])

    def test_selected_root_wins_over_still_existing_old_source(self):
        old = original(self.base / 'old' / 'iTERO' / 'G3' / 'BCG3T0-1')
        selected = original(self.raw / 'iTERO' / 'G3-OK' / 'BCG3T0-1')
        self.metadata(source_ply=str(old))
        mesh = OriginalModels(self.root, self.raw).load(self.scan)
        self.assertEqual(mesh['loaded_path'], str(selected.resolve()))

    def test_ambiguous_models_are_not_guessed(self):
        folder = self.raw / 'iTERO' / 'G3' / 'BCG3T0-1'
        original(folder, 'first.ply'); original(folder, 'second.ply')
        resolver = OriginalModels(self.root, self.raw)
        with self.assertRaisesRegex(ValueError, 'Ambiguous'): resolver.load(self.scan)
        self.metadata(source_ply=r'C:\old\iTERO\G3\BCG3T0-1\second.ply')
        self.assertTrue(resolver.load(self.scan)['loaded_path'].endswith('second.ply'))

    def test_missing_original_does_not_substitute_another_stage(self):
        original(self.raw / 'iTERO' / 'G3' / 'BCG3T1-1')
        with self.assertRaisesRegex(FileNotFoundError, 'Original missing'): OriginalModels(self.root, self.raw).load(self.scan)

    def test_full_texture_uv_is_loaded_without_vertex_colour(self):
        original(self.raw / 'iTERO' / 'G3' / 'BCG3T0-1', textured=True)
        mesh = OriginalModels(self.root, self.raw).load(self.scan)
        self.assertIsNotNone(mesh['texture'])
        self.assertIsNotNone(mesh['poly'].GetPointData().GetTCoords())
        self.assertEqual(mesh['association'], 'none')
        self.assertTrue(mesh['texture_path'].endswith('full_texture.png'))


if __name__ == '__main__': unittest.main()
