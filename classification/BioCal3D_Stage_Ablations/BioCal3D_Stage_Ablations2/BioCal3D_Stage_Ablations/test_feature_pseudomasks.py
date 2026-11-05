"""Synthetic integration checks; no checkpoints, patient data, or GPU needed.

Run beside the stage-ablation modules: python -m unittest test_feature_pseudomasks -v
"""
import contextlib
import csv
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import joblib
import numpy as np
from PIL import Image

import make_feature_pseudomasks as p
from study_data import digest, signature, write_json


def fixture(root):
    rows = []
    for mode in ("dinov3", "concerto"):
        write_json(root / "features" / f"{mode}_config.json", {"encoder": mode, "synthetic": True})
    for partition, fold, specimen in (("development", 0, "val"), ("development", 1, "train"), ("test", None, "test")):
        for scanner in ("LABscanner", "TRIOS3", "TRIOS5", "iTERO"):
            for stage in p.STAGES:
                rid = f"{scanner}_{specimen}_{stage}"
                rows.append(dict(row_id=rid, scan_id=rid, physical_id=specimen, partition=partition,
                                 fold=fold, scanner=scanner, stage=stage, sample_name=rid, geometry_sha256="synthetic"))
    write_json(root / "cohort.json", {"rows": rows})
    yy, xx = np.mgrid[:8, :8]
    xyz = np.column_stack(((xx.ravel() + .5) * 2 - 8, 8 - (yy.ravel() + .5) * 2, np.zeros(64))).astype(np.float32)
    for row in rows:
        # Test caches deliberately absent: fit and validation must not access them.
        if row["partition"] == "test":
            continue
        rng = np.random.default_rng(p.seed_for(row["scan_id"]))
        n = {"baseline": 48, "partial": 28, "clean": 2}[row["stage"]]
        plaque = np.zeros(64, bool); plaque[:n] = True
        x = rng.normal(0, .04, (64, 12)).astype(np.float32)
        x[:, 0] += np.where(plaque, 2., -2.)
        x[:, 1] += np.where(plaque, -1., 1.)
        x[:, 2] += 1
        folder = root / "renders" / "development" / "images" / row["scanner"] / row["sample_name"]
        folder.mkdir(parents=True)
        rgb = np.full((64, 64, 3), [200, 95, 110], np.uint8)
        Image.fromarray(rgb).save(folder / "rgb.png")
        Image.fromarray(np.full((64, 64), 255, np.uint8)).save(folder / "valid_mask.png")
        np.savez(folder / "maps.npz", source_geometry_sha256="synthetic", center_mm=np.zeros(3), field_mm=16.)
        for mode in ("dinov3_rgb", "concerto_rgb"):
            config = p.read_json(root / "features" / f"{mode.split('_')[0]}_config.json")
            key = signature([signature(config), row["geometry_sha256"], row["row_id"]])
            file = p.feature_path(root, mode, row)
            file.parent.mkdir(parents=True, exist_ok=True)
            if mode.startswith("dinov3"):
                local = x.T.reshape(12, 8, 8)
                np.savez(file, local=local, coverage=np.ones((8, 8), np.float32), key=key)
            else:
                np.savez(file, local=x, xyz_scaled=xyz, scale_mm=1., rotation=np.eye(3),
                         center_mm=np.zeros(3), face_index=np.zeros(64, int),
                         barycentric=np.tile([1., 0, 0], (64, 1)), key=key)
    return rows


class PseudoMasks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.dataset = cls.root / "run"
        cls.rows = fixture(cls.dataset)
        cls.model = cls.root / "model"
        cls.fitargs = SimpleNamespace(run=cls.dataset, model_dir=cls.model,
            modes=["dinov3_rgb", "concerto_rgb"], validation_fold=0, clusters=4,
            pca_dimensions=6, samples_per_scan=32, min_coverage=.5, seeds_csv=None, seed=123)
        with contextlib.redirect_stdout(io.StringIO()):
            p.train(cls.fitargs)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def predargs(self, out, **changes):
        values = dict(run=self.dataset, model_dir=self.model, out=self.root / out,
                      partition="validation", modes=None, scan=None, scanner=None,
                      limit=None, low=.35, high=.65)
        values.update(changes)
        return SimpleNamespace(**values)

    def test_end_to_end_two_modes_four_scanners(self):
        args = self.predargs("predictions")
        with contextlib.redirect_stdout(io.StringIO()):
            p.predict(args)
        with (args.out / "summary.csv").open(encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 24)
        self.assertEqual(len({r["scanner"] for r in rows}), 4)
        self.assertTrue(all(abs(sum(float(r[k]) for k in ("plaque_like_fraction", "uncertain_fraction", "clean_like_fraction")) - 1) < 1e-6 for r in rows))
        self.assertEqual(len(list(args.out.glob("*/comparison.png"))), 12)
        self.assertEqual(len(list(args.out.glob("*/concerto_rgb/pseudo_points.ply"))), 12)
        scores = {s: np.mean([float(r["mean_pseudo_score"]) for r in rows if r["stage"] == s]) for s in p.STAGES}
        self.assertGreater(scores["baseline"], scores["clean"])
        for file in args.out.glob("*/*/pseudo_mask.png"):
            self.assertTrue(np.isin(np.array(Image.open(file)), [0, 128, 255]).all())

    def test_training_does_not_use_validation_or_test(self):
        m = joblib.load(self.model / "dinov3_rgb.joblib")
        self.assertEqual(m["training_physical_ids"], ["train"])
        self.assertTrue(all("_train_" in s for s in m["training_scan_ids"]))
        self.assertFalse((self.dataset / "features" / "dinov3_rgb" / "test").exists())

    def test_stale_cache_rejected(self):
        row = next(r for r in self.rows if r["partition"] == "development")
        wrong = dict(row, geometry_sha256="changed")
        with self.assertRaisesRegex(ValueError, "mismatched"):
            p.load_local(self.dataset, wrong, "dinov3_rgb", .5)

    def test_validation_seeds_rejected(self):
        file = self.root / "bad_seeds.csv"
        file.write_text("scan_id,mode,path\nLABscanner_val_baseline,dinov3_rgb,unused.png\n")
        with self.assertRaisesRegex(ValueError, "outside the training"):
            p.load_seed_table(file, [r for r in self.rows if r["physical_id"] == "train"])

    def test_shared_pca_and_round_robin_selection(self):
        args = self.predargs("preview", limit=4)
        with contextlib.redirect_stdout(io.StringIO()):
            p.predict(args)
        with (args.out / "summary.csv").open(encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        self.assertGreater(len({r["scanner"] for r in rows}), 1)

    def test_weak_seed_fit_both_modalities(self):
        row = next(r for r in self.rows if r["physical_id"] == "train" and r["stage"] == "partial")
        x, eligible, _, meta = p.load_local(self.dataset, row, "concerto_rgb", .5)
        labels = (x[:, 0] > 0).astype(np.int8)
        point_file = self.root / "point_seeds.npz"
        np.savez(point_file, labels=labels, xyz_mm=meta["xyz_mm"])
        image_file = self.root / "image_seeds.png"
        mask = np.repeat(np.repeat(labels.reshape(8, 8).astype(np.uint8) * 255, 8, axis=0), 8, axis=1)
        Image.fromarray(mask).save(image_file)
        table = self.root / "seeds.csv"
        table.write_text("scan_id,mode,path\n" + f"{row['scan_id']},dinov3_rgb,image_seeds.png\n{row['scan_id']},concerto_rgb,point_seeds.npz\n")
        args = SimpleNamespace(**vars(self.fitargs))
        args.model_dir = self.root / "seeded_model"
        args.seeds_csv = table
        with contextlib.redirect_stdout(io.StringIO()):
            p.train(args)
        for mode in args.modes:
            self.assertEqual(joblib.load(args.model_dir / f"{mode}.joblib")["method"], "weak_seed_classifier")
        pred = self.predargs("seeded_prediction", model_dir=args.model_dir, limit=1)
        with contextlib.redirect_stdout(io.StringIO()):
            p.predict(pred)

    def test_unknown_scan_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown scan"):
            p.predict(self.predargs("bad", scan=["not-a-scan"]))

    def test_thresholds_retain_uncertainty(self):
        np.testing.assert_array_equal(p.native_labels(np.array([.1, .5, .9]), .35, .65), [0, -1, 1])


if __name__ == "__main__":
    unittest.main()
