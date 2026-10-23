import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import investigate_approved_rois as app
from make_fixture import make_dataset


def square_mesh():
    return np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]]), np.array([[0, 1, 2], [1, 3, 2]])


class GeometryTests(unittest.TestCase):
    def test_tilt_and_translation_are_removed(self):
        v, f = square_mesh(); v[:, 2] = .2*v[:, 0]-.1*v[:, 1]+2; v += [100., -30., 10.]
        r, _, _ = app.mesh_measurements(v, f, 2000)
        self.assertLess(r["height_rms_um"], 1e-4)
        self.assertLess(r["height_mean_abs_um"], 1e-6)
        self.assertAlmostEqual(r["surface_area_mm2"], np.sqrt(1.05), places=9)

    def test_exact_rms_matches_independent_dense_sampling(self):
        v, f = square_mesh(); v[3, 2] = .2
        r, xyz, area = app.mesh_measurements(v, f, 2000)
        rng = np.random.default_rng(13); ids = rng.choice(len(f), 400000, p=area/area.sum())
        u = rng.random((len(ids), 2)); take = u.sum(1) > 1; u[take] = 1-u[take]
        h = xyz[f[ids], 2]; z = h[:, 0]*(1-u.sum(1))+h[:, 1]*u[:, 0]+h[:, 2]*u[:, 1]
        self.assertAlmostEqual(r["height_rms_um"], np.sqrt(np.mean(z*z))*1000, delta=.1)

    def test_uv_seam_duplicates_do_not_change_area_or_roughness(self):
        v, f = square_mesh(); v[3, 2] = .1
        r1, _, _ = app.mesh_measurements(v, f, 2000)
        v2 = np.r_[v, v[[1, 2]]]; f2 = np.array([[0, 1, 2], [4, 3, 5]])
        r2, _, _ = app.mesh_measurements(v2, f2, 2000)
        self.assertEqual(r2["vertex_count"], 6); self.assertEqual(r2["geometric_vertex_count"], 4)
        for key in ["surface_area_mm2", "height_rms_um", "height_mean_abs_um", "edge_median_um"]:
            self.assertAlmostEqual(r1[key], r2[key], places=9)

    def test_common_grid_removes_plane_and_preserves_holes(self):
        x = (np.arange(100)+.5)*.01-.5; xx, yy = np.meshgrid(x, x)
        h = .1*xx+.05*yy+1; valid = np.ones_like(h, bool); valid[40:60, 40:60] = False
        area = np.full_like(h, .0001)
        r, grid = app.common_grid(h, valid, area, 1., .1)
        self.assertLess(r["height_rms_um"], 1e-8)
        self.assertTrue(np.isnan(grid[4:6, 4:6]).all())
        with self.assertRaises(ValueError):
            app.common_grid(h, valid, area, 1., .001)

    def test_shared_scale_averaging_attenuates_high_frequency(self):
        x = (np.arange(200)+.5)*.005- .5; xx, yy = np.meshgrid(x, x)
        h = .01*np.sin(2*np.pi*xx/.05); area = np.full_like(h, .005**2); valid = np.ones_like(h, bool)
        fine, _ = app.common_grid(h, valid, area, 1., .01)
        coarse, _ = app.common_grid(h, valid, area, 1., .1)
        self.assertLess(coarse["height_rms_um"], fine["height_rms_um"]*.05)

    def test_noninteger_pixel_cell_ratio_has_no_false_holes_or_roughness(self):
        coordinates = (np.arange(64)+.5)*2/64-1
        xx, yy = np.meshgrid(coordinates, coordinates)
        h = .13*xx-.07*yy+.4; valid = np.ones_like(h, bool)
        area = np.full_like(h, (2/64)**2)
        r, grid = app.common_grid(h, valid, area, 2., .05)
        self.assertEqual(grid.shape, (40, 40)); self.assertTrue(np.isfinite(grid).all())
        self.assertLess(r["height_rms_um"], 1e-8)
        self.assertAlmostEqual(r["visible_projected_area_mm2"], 4., places=10)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.base = Path(self.temp.name)
        self.root = make_dataset(self.base/"approved", specimens=2, scanners=("iTERO", "LABscanner"))

    def tearDown(self):
        self.temp.cleanup()

    def args(self):
        return app.parse_args(["--data-root", str(self.root), "--surface-samples", "2000", "--bootstrap", "100", "--dpi", "90"])

    def test_protocol_and_pairing(self):
        inventory = app.discover(self.root)
        self.assertEqual(len(inventory), 20)
        self.assertEqual(set(inventory.loc[(inventory.group == 1)&(inventory.timepoint == "T1"), "stage"]), {"clean"})
        self.assertEqual(set(inventory.loc[(inventory.group == 3)&(inventory.timepoint == "T1"), "stage"]), {"partial"})
        data, cache = [], {}
        for row in inventory.to_dict("records"):
            stats, arrays = app.load_and_measure(row, self.args()); data.append(stats); cache[row["scan_id"]] = arrays
        paired, stage = app.paired_tables(pd.DataFrame(data), cache, self.args())
        self.assertEqual(len(paired), 10)
        self.assertEqual(len(stage), 16)  # 4 fresh baseline-clean and 12 mature transitions.
        self.assertTrue((paired.deltaE00_mean_Lab > 0).all())
        self.assertTrue((stage.delta_L_mean > 0).all())

    def test_mapping_mismatch_is_rejected(self):
        row = app.discover(self.root).iloc[0].to_dict(); p = Path(row["sample_dir"])/"maps.npz"
        with np.load(p) as m:
            data = {k: m[k] for k in m.files}
        data["vertices"] = data["vertices"]+.001; np.savez_compressed(p, **data)
        with self.assertRaisesRegex(ValueError, "Maps do not match"):
            app.load_and_measure(row, self.args())

    def test_saved_texture_map_takes_precedence_over_vertex_colours(self):
        row = app.discover(self.root).iloc[0].to_dict(); folder = Path(row["sample_dir"])
        p = folder/"geometry.npz"
        with np.load(p) as g:
            values = {k:g[k] for k in g.files}
        values["vertex_rgb"] = np.tile([0., 1., 0.], (len(values["vertices"]), 1))
        np.savez_compressed(p, **values)
        meta = json.loads((folder/"metadata.json").read_text()); meta["color_source"] = "texture_uv"
        (folder/"metadata.json").write_text(json.dumps(meta))
        stats, _ = app.load_and_measure(row, self.args())
        self.assertEqual(stats["color_source"], "texture_uv")
        self.assertGreater(stats["a_mean"], 0)  # The saved pink texture map wins, not fake green vertices.

    def test_colour_uses_area_weights_and_excludes_background(self):
        row = app.discover(self.root).iloc[0].to_dict(); p = Path(row["sample_dir"])/"maps.npz"
        with np.load(p) as m:
            data = {k: m[k] for k in m.files}
        valid = data["valid"]; ids = np.flatnonzero(valid); half = len(ids)//2
        data["rgb"][:] = [1., 0., 0.]; data["rgb"].reshape(-1, 3)[ids[:half]] = 0
        data["rgb"].reshape(-1, 3)[ids[half:]] = 1
        data["pixel_surface_area_mm2"].flat[ids[:half]] = 1
        data["pixel_surface_area_mm2"].flat[ids[half:]] = 3
        np.savez_compressed(p, **data)
        stats, _ = app.load_and_measure(row, self.args())
        expected = 100*3*(len(ids)-half)/(half+3*(len(ids)-half))
        self.assertAlmostEqual(stats["L_mean"], expected, places=3)

    def test_split_leakage_and_authoritative_specimen_csv(self):
        row = app.discover(self.root).iloc[0]; p = Path(row.sample_dir)/"metadata.json"
        meta = json.loads(p.read_text()); meta["split"] = "val"; p.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            app.discover(self.root)
        csv = self.base/"splits.csv"
        csv.write_text("physical_id,split\nG1_S1,train\nG1_S2,test\nG3_S1,train\nG3_S2,test\n")
        result = app.discover(self.root, csv)
        self.assertEqual(set(result.loc[result.physical_id == "BCG1-1", "split"]), {"train"})

    def test_ciede2000_reference_pair(self):
        value = app.deltaE_ciede2000(np.array([[50., 2.6772, -79.7751]]), np.array([[50., 0., -82.7485]]))[0]
        self.assertAlmostEqual(value, 2.0425, places=4)

    def test_cli_all_and_training_scope_and_resume(self):
        hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob("*") if p.is_file()}
        out = self.base/"all"
        cmd = ["--data-root", str(self.root), "--output", str(out), "--tables-only", "--surface-samples", "2000", "--bootstrap", "100"]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(app.main(cmd), 0)
            with mock.patch.object(app, "load_and_measure", side_effect=RuntimeError("Cache not used")):
                self.assertEqual(app.main(cmd+["--resume"]), 0)
        self.assertEqual(len(pd.read_csv(out/"tables"/"scan_measurements.csv")), 20)
        summary = pd.read_csv(out/"tables"/"scanner_stage_summary.csv")
        self.assertTrue({"L_sd", "vertex_count"} <= set(summary.metric))
        self.assertTrue((out/"tables"/"common_scale_summary.csv").is_file())
        with self.assertRaisesRegex(ValueError, "scope"):
            app.main(cmd+["--resume", "--scope", "train"])
        train = self.base/"train"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(app.main(["--data-root", str(self.root), "--output", str(train), "--scope", "train", "--tables-only", "--surface-samples", "2000", "--bootstrap", "100"]), 0)
        selected = pd.read_csv(train/"tables"/"scan_measurements.csv")
        self.assertEqual(len(selected), 10); self.assertEqual(set(selected.specimen_number), {1})
        self.assertEqual(hashes, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob("*") if p.is_file()})
        self.assertTrue((out/"report.html").is_file())


if __name__ == "__main__":
    unittest.main()
