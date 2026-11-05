"""Synthetic validation of exact head explanations and intervention bookkeeping."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys
import contextlib
import io

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import explain_stage_classifier as e
from test_feature_pseudomasks import fixture


class StageExplanation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.dataset = cls.root / "run"
        cls.rows = fixture(cls.dataset)
        for mode in ("dinov3_rgb", "concerto_rgb"):
            training, labels = [], []
            for row in cls.rows:
                file = e.feature_path(cls.dataset, mode, row)
                if not file.exists(): continue
                with np.load(file) as z:
                    arrays = {k:z[k] for k in z.files}
                local = arrays["local"]
                if mode.startswith("dinov3"):
                    x=local.reshape(local.shape[0],-1).T
                    w=arrays["coverage"].ravel(); eligible=w>=.05
                else:
                    x=local;w=np.ones(len(x));eligible=np.ones(len(x),bool)
                arrays["pooled"] = e.pool(x,w,eligible).astype(np.float32)
                np.savez(file,**arrays)
                if row["physical_id"]=="train":
                    training.append(arrays["pooled"]);labels.append(e.STAGES.index(row["stage"]))
            model=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight="balanced",max_iter=3000)).fit(training,labels)
            target=cls.dataset / "locked_test" / e.RESULT_SUBDIR / f"{mode}_all_development_head.joblib"
            target.parent.mkdir(parents=True,exist_ok=True);joblib.dump(model,target)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def row(self):
        return next(r for r in self.rows if r["physical_id"]=="val" and r["stage"]=="partial")

    def test_exact_contributions_and_spatial_allocation(self):
        for mode in ("dinov3_rgb","concerto_rgb"):
            data=e.load_cache(self.dataset,self.row(),mode)
            head,_=e.load_head(self.dataset,mode,"locked_test")
            result=e.decomposition(head,data,1,2)
            self.assertAlmostEqual(result["terms"].sum()+result["intercept"],result["margin"],places=5)
            self.assertAlmostEqual(result["combined"].sum()+result["intercept"],result["margin"],places=5)

    def test_replacing_feature_at_training_mean_matches_exact_term(self):
        data=e.load_cache(self.dataset,self.row(),"dinov3_rgb")
        head,_=e.load_head(self.dataset,"dinov3_rgb","locked_test")
        result=e.decomposition(head,data,1,2)
        for j in range(len(result["terms"])):
            changed=data["pooled"].copy();changed[j]=head.named_steps["standardscaler"].mean_[j]
            self.assertAlmostEqual(result["margin"]-e.margin(head,changed,1,2),result["terms"][j],places=6)

    def test_feature_region_deletion_matches_direct_pooling(self):
        data=e.load_cache(self.dataset,self.row(),"dinov3_rgb")
        head,_=e.load_head(self.dataset,"dinov3_rgb","locked_test")
        tiles=np.repeat(np.arange(8),8)
        heat,records=e.region_deletions(head,data,tiles,8,1,2)
        self.assertEqual(len(records),8)
        expected=e.margin(head,data["pooled"],1,2)-e.margin(head,e.pool(data["x"],data["w"],data["eligible"],tiles!=0),1,2)
        self.assertAlmostEqual(heat.ravel()[0],expected,places=6)

    def test_cached_figures_and_signed_output(self):
        out=self.root / "cached_output";out.mkdir()
        args=SimpleNamespace(run=self.dataset,modes=["dinov3_rgb","concerto_rgb"],target="partial",against="clean",
                             head_source="locked_test",grid=4,top=8)
        e.cached(args,self.row(),out)
        self.assertTrue((out / "classifier_evidence.png").is_file())
        for mode in args.modes:
            for name in ("feature_contributions.csv","decision_features.png","top_channel_locations.png","feature_stage_distributions.png","allocation_detail.png","exact_evidence.npz"):
                self.assertTrue((out/mode/name).is_file(),name)

    def test_input_intervention_reencodes_each_region_and_fill(self):
        row=self.row();data=e.load_cache(self.dataset,row,"dinov3_rgb")
        calls=[]
        def encode(selected=None,fill="median"):
            calls.append((selected is None,fill))
            changed=data["pooled"].copy()
            if selected is not None:changed[0]-=float(selected.mean())*(.5 if fill=="median" else 1)
            return changed
        args=SimpleNamespace(run=self.dataset,target="partial",against="clean",head_source="locked_test",grid=2,
                             encoder="dinov3",mode="dinov3_rgb",intervention="appearance")
        out=self.root / "input_output";out.mkdir()
        with patch.object(e,"make_reencoder",return_value=encode):
            e.input_occlusion(args,row,out)
        self.assertEqual(sum(c[0] for c in calls),1)
        self.assertEqual(len(calls),9)
        self.assertTrue((out / "input_occlusion.png").is_file())
        with np.load(out / "input_occlusion.npz") as z:
            self.assertFalse(np.allclose(z["median"],z["neutral"]))

    def test_dino_reencoder_edits_original_roi_pixels(self):
        from PIL import Image
        import models_2d
        row=self.row();data=e.load_cache(self.dataset,row,"dinov3_rgb")
        weight=self.root / "fake_checkpoint.pth";weight.write_bytes(b"synthetic fixture")
        data["config"]=dict(data["config"],checkpoint_sha256=e.digest(weight),upstream_commit="synthetic-commit",encoder_state_sha256="synthetic-state")
        seen=[]
        class FakeEncoder:
            def __init__(self,*args,**kwargs):self.meta={"encoder_state_sha256":"synthetic-state"}
            def extract(self,image,mask):
                seen.append((image.copy(),mask.copy()))
                return np.array([image.mean()]),None,None
        args=SimpleNamespace(encoder="dinov3",checkpoint=weight,repo=self.root / "unused_repo",device="cpu",run=self.dataset)
        with patch.dict(sys.modules,{"torch":SimpleNamespace()}),patch.object(models_2d,"Encoder",FakeEncoder),patch.object(models_2d,"commit",return_value="synthetic-commit"):
            encode=e.make_reencoder(args,row,data,"dinov3_rgb")
            encode()
            selected=np.zeros((64,64),bool);selected[:8,:8]=True
            encode(selected,"neutral")
        self.assertTrue((seen[1][0][selected]==127).all())
        np.testing.assert_array_equal(seen[1][0][~selected],seen[0][0][~selected])
        np.testing.assert_array_equal(seen[1][1],seen[0][1])

    def selection_args(self,**changes):
        values=dict(scan=None,partition=None,stage=None,scanner=None,specimen=None,limit=None,all=False,list_scans=False,target=None,against=None)
        values.update(changes)
        return SimpleNamespace(**values)

    def test_batch_defaults_test_and_balances_scanner_stage(self):
        rows=e.choose_scans(self.rows,self.selection_args())
        self.assertEqual(len(rows),12)
        self.assertTrue(all(r["partition"]=="test" for r in rows))
        self.assertEqual(len({(r["scanner"],r["stage"]) for r in rows}),12)

    def test_batch_filters_and_explicit_ids(self):
        rows=e.choose_scans(self.rows,self.selection_args(partition="development",stage=["partial"],scanner=["iTERO"],all=True))
        self.assertEqual(len(rows),2)
        explicit=[rows[1]["scan_id"],rows[0]["scan_id"]]
        selected=e.choose_scans(self.rows,self.selection_args(scan=explicit))
        self.assertEqual([r["scan_id"] for r in selected],explicit)
        with self.assertRaisesRegex(ValueError,"conflict"):
            e.choose_scans(self.rows,self.selection_args(scan=explicit,partition="test"))

    def test_auto_contrasts_and_specimen_selection(self):
        clean=next(r for r in self.rows if r["stage"]=="clean")
        self.assertEqual(e.comparison_for(clean,self.selection_args()),("clean","partial"))
        rows=e.choose_scans(self.rows,self.selection_args(partition="all",specimen=["val"],all=True))
        self.assertEqual(len(rows),12)

    def test_batch_command_creates_index_and_combined_summary(self):
        from PIL import Image
        scans=[r["scan_id"] for r in self.rows if r["physical_id"]=="val"][:2]
        output=self.root / "batch_command"
        def fake_cached(args,row,out):
            Image.new("RGB",(32,32),"gray").save(out / "classifier_evidence.png")
            return [dict(scan_id=row["scan_id"],mode="dinov3_rgb",margin=1,target_stage=args.target,comparison_stage=args.against)]
        argv=["explain_stage_classifier.py","cached","--run",str(self.dataset),"--scan",*scans,"--out",str(output)]
        with patch.object(sys,"argv",argv),patch.object(e,"cached",side_effect=fake_cached),contextlib.redirect_stdout(io.StringIO()):e.main()
        self.assertTrue((output / "index.html").is_file())
        self.assertEqual((output / "batch_decisions.csv").read_text(encoding="utf-8-sig").count("dinov3_rgb"),2)
        self.assertEqual(len(list(output.glob("*/classifier_evidence.png"))),2)
        self.assertIn(scans[0],(output / "index.html").read_text())


if __name__=="__main__":unittest.main()
