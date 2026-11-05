"""Synthetic checks for exact cosine, native coordinates, prompts and exports."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

import inspect_feature_maps as f
from test_feature_pseudomasks import fixture
from stage_ablation import read_json


class FeatureInspector(unittest.TestCase):
    def test_cosine_matches_normalized_dot_product(self):
        x = np.array([[1., 0], [0., 2], [1., 1], [-1., 0]])
        scores = f.reference_similarities(x, x, [np.array([0])], [np.array([1])])
        np.testing.assert_allclose(scores["positive"], [1, 0, 2 ** -.5, -1], atol=1e-6)
        np.testing.assert_allclose(scores["negative"], [0, 1, 2 ** -.5, 0], atol=1e-6)
        np.testing.assert_allclose(scores["margin"], scores["positive"] - scores["negative"])
        np.testing.assert_array_equal(f.prompt_labels(scores["margin"], .02), [1, 0, -1, 0])

    def test_multiple_positive_appearances(self):
        x = np.eye(3)
        scores = f.reference_similarities(x, x, [np.array([0]), np.array([1])], [])
        np.testing.assert_array_equal(scores["positive"], [1, 1, 0])
        self.assertNotIn("margin", scores)

    def test_physical_reference_radius(self):
        xy = np.array([[10., 10.], [11., 10.], [50., 50.]])
        np.testing.assert_array_equal(f.choose_region(xy, [10, 10], .2, (100, 100), 10), [0, 1])
        np.testing.assert_array_equal(f.choose_region(xy, [10, 10], 0, (100, 100), 10), [0])

    def test_pca_variance_and_later_components(self):
        x = np.random.default_rng(1).normal(size=(50, 12))
        coords, variance = f.per_scan_pca(x)
        self.assertTrue(np.all(np.diff(variance) <= 0))
        for start in (0, 3, 6):
            rgb = f.pca_rgb(coords, start)
            self.assertEqual(rgb.shape, (50, 3))
            self.assertTrue(((rgb >= 0) & (rgb <= 1)).all())
        self.assertFalse(np.allclose(f.pca_rgb(coords), f.pca_rgb(coords, 3)))

    def test_viewer_callbacks_and_native_exports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = fixture(root / "run")
            row = next(r for r in rows if r["partition"] == "development" and r["stage"] == "partial")
            args = SimpleNamespace(run=root / "run", modes=["dinov3_rgb", "concerto_rgb"],
                radius_mm=0., margin=.02, pc_start=1, fixed_scale=False,
                min_coverage=.5, clusters=4, pca_style="components", out=root / "exports")
            viewer = f.Inspector(args, row, row)
            self.assertTrue(viewer.add_click(3.5, 3.5, "positive"))
            self.assertFalse(viewer.add_click(3.5, 3.5, "negative"))
            self.assertTrue(viewer.add_click(51.5, 59.5, "negative"))
            viewer.change_pc(4)
            viewer.change_margin(.01)
            viewer.toggle_scale()
            out = viewer.save()
            for mode in args.modes:
                folder = out / mode
                self.assertTrue((folder / "maps.png").is_file())
                self.assertTrue((folder / "pca_pc4_6.png").is_file())
                self.assertTrue((folder / "positive_cosine.tiff").is_file())
                with np.load(folder / "native_maps.npz") as z:
                    self.assertTrue(np.isfinite(z["positive"]).all())
                    self.assertTrue(np.isin(z["pseudo_label"], [-1, 0, 1]).all())
                    self.assertEqual(z["pca_variance"].shape[0], 9)
                self.assertTrue(np.isin(np.array(Image.open(folder / "prompted_mask.png")), [0, 128, 255]).all())
            self.assertTrue((out / "concerto_rgb" / "prompted_segmentation_points.ply").is_file())
            replay = read_json(out / "prompts.json")
            self.assertEqual(len(replay["clicks"]), 2)
            viewer.undo(); self.assertEqual(len(viewer.clicks), 1)
            viewer.clear(); self.assertEqual(len(viewer.clicks), 0)
            f.Inspector(args, row, next(r for r in rows if r["partition"] == "development" and r["stage"] == "clean"))
            viewer.plt.close("all")


if __name__ == "__main__":
    unittest.main()
