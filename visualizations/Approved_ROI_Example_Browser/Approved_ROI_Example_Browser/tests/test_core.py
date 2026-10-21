"""Functional pairing, colour and data-protection checks; no GUI required."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import roi_browser_core as core


def sample(root, scanner='iTERO', group=3, timepoint=0, specimen=1, size=1):
    folder = root / 'samples' / scanner / f'G{group}' / f'BCG{group}T{timepoint}-{specimen}'
    folder.mkdir(parents=True, exist_ok=True)
    vertices = np.array([[0., 0, 0], [size, 0, 0], [0, size, 0], [size, size, 0]])
    np.savez(folder / 'geometry.npz', vertices=vertices,
             faces=np.array([[0, 1, 2], [1, 3, 2]]),
             rgb=np.array([[230, 10, 20], [10, 230, 20], [10, 20, 230], [230, 230, 20]], np.uint8))
    return folder / 'geometry.npz'


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.base = Path(self.temp.name)
        self.root = self.base / 'Approved ROIs'; self.root.mkdir()
    def tearDown(self): self.temp.cleanup()

    def test_complete_48_specimen_512_scan_design_includes_every_group(self):
        for group in range(1, 7):
            for number in range(1, 9):
                for timepoint in range(2 if group <= 2 else 3):
                    for scanner in core.SCANNERS: sample(self.root, scanner, group, timepoint, number)
        root, scans, warnings = core.discover(self.root)
        self.assertEqual(len(scans), 512); self.assertEqual(len({s.physical_id for s in scans}), 48)
        self.assertFalse(warnings); self.assertEqual(sum(s.group == 5 for s in scans), 96)
        self.assertEqual(sum(s.stage == 'partial' for s in scans), 128)
        self.assertEqual(sum(s.stage == 'clean' for s in scans), 192)
        self.assertEqual(sum(s.stage == 'baseline' for s in scans), 192)

    def test_incomplete_pairs_never_substitute_another_specimen(self):
        sample(self.root, 'iTERO', 3, 1, 1); sample(self.root, 'TRIOS3', 3, 1, 2)
        _, scans, _ = core.discover(self.root)
        lookup = {s.key: s for s in scans}
        self.assertIn(('BCG3-1', 'partial', 'iTERO'), lookup)
        self.assertNotIn(('BCG3-1', 'partial', 'TRIOS3'), lookup)

    def test_two_stage_design_and_review_folders(self):
        path = sample(self.root, group=1, timepoint=1)
        review = path.parent / 'roi.vtp' / 'versions' / 'old'; review.mkdir(parents=True)
        (review / 'surface.npz').write_bytes(b'not an input')
        _, scans, _ = core.discover(self.root)
        self.assertEqual(len(scans), 1); self.assertEqual(scans[0].stage, 'clean')

    def test_duplicate_identity_fails(self):
        sample(self.root, 'iTERO'); sample(self.root, 'ITERO')
        with self.assertRaisesRegex(ValueError, 'Duplicate'): core.discover(self.root)

    def test_export_cannot_write_inside_dataset(self):
        for output in (self.root, self.root / 'exports'):
            with self.assertRaises(ValueError): core.safe_output(self.root, output)
        self.assertFalse((self.root / 'exports').exists())

    def test_unmodified_rgb_and_mask_with_centered_display(self):
        from vtk.util.numpy_support import vtk_to_numpy
        path = sample(self.root)
        with np.load(path) as data: values = {key: data[key] for key in data.files}
        values['face_roi_mask'] = [True, False]; np.savez(path, **values)
        before = path.read_bytes(); scan = core.discover(self.root)[1][0]
        mesh = core.load_mesh(scan, self.root)
        self.assertEqual(mesh['faces'], 1)
        np.testing.assert_array_equal(vtk_to_numpy(mesh['poly'].GetPointData().GetScalars()), values['rgb'])
        np.testing.assert_allclose(mesh['center_translation'], [.5, .5, 0])
        self.assertEqual(before, path.read_bytes())

    def test_float_rgb_and_invalid_mesh(self):
        np.testing.assert_array_equal(core.rgb_array(np.array([[1., .5, 0]]), 1), [[255, 128, 0]])
        path = sample(self.root)
        np.savez(path, vertices=np.zeros((3, 3)), faces=[[0, 1, 8]])
        with self.assertRaisesRegex(ValueError, 'Invalid'): core.load_mesh(core.discover(self.root)[1][0], self.root)

    def test_texture_does_not_multiply_vertex_rgb(self):
        import vtk
        path = sample(self.root); mesh = core.load_mesh(core.discover(self.root)[1][0], self.root)
        mesh['texture'] = vtk.vtkTexture()
        actor = core.actor_for(mesh)
        self.assertFalse(actor.GetMapper().GetScalarVisibility())
        self.assertEqual(actor.GetProperty().GetColor(), (1, 1, 1))

    def test_inventory_lists_physical_identity_and_stage(self):
        sample(self.root, group=2, timepoint=1, specimen=8)
        path = self.base / 'inventory.csv'; core.write_inventory(core.discover(self.root)[1], path)
        text = path.read_text(encoding='utf-8-sig')
        self.assertIn('BCG2-8,BCG2T1-8,2,8,iTERO,clean,1', text)


if __name__ == '__main__': unittest.main()
