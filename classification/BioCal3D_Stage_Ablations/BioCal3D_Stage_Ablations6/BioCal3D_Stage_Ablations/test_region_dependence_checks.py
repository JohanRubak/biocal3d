"""Synthetic checks for region matching, original-input edits and specimen summaries."""
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import test_plaque_dependence as d
import test_stage_explanations as fixtures


class DependenceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.StageExplanation.setUpClass()
        cls.dataset=fixtures.StageExplanation.dataset
        cls.root=fixtures.StageExplanation.root
        cls.row=next(r for r in fixtures.StageExplanation.rows if r["physical_id"]=="val" and r["stage"]=="partial")
        # Add visible contrast for the mock encoder; the real encoders are not used here.
        folder=cls.dataset/"renders"/"development"/"images"/cls.row["scanner"]/cls.row["sample_name"]
        rgb=np.array(Image.open(folder/"rgb.png"))
        rgb[:32]=[160,60,80];rgb[32:]=[200,95,110]
        Image.fromarray(rgb).save(folder/"rgb.png")
        cls.pair=dict(pair_id=1,radius_mm=1.,plaque_xy=[11.5,11.5],clean_xy=[51.5,51.5])
        cls.selection=cls.root/"regions.json"
        d.write_json(cls.selection,d.annotation(cls.dataset,cls.row,[cls.pair]))
        cls.weight=cls.root/"mock_checkpoint.pth";cls.weight.write_bytes(b"mock encoder fixture")

    @classmethod
    def tearDownClass(cls):fixtures.StageExplanation.tearDownClass()

    def args(self,encoder="dinov3",**updates):
        values=dict(run=self.dataset,encoder=encoder,mode=None,target="partial",against="clean",
            head_source="locked_test",intervention="appearance",fills=["median","neutral"],
            random_controls=2,rim_tolerance=.75,min_units=1,seed=123,checkpoint=self.weight,
            repo=self.root/"unused_repo",snapshot=None,device="cpu")
        values.update(updates);return SimpleNamespace(**values)

    def mock_factory(self,args,row,data,mode):
        rgb,valid,_,field=d.render(args.run,row)
        if args.encoder=="dinov3":
            source=rgb
            def inspect(region=None,fill="median"):
                changed=source.copy()
                if region is not None:
                    changed[region]=np.rint(np.median(source[valid],axis=0) if fill=="median" else [127,127,127]).astype(np.uint8)
                return dict(image=changed,mask=valid.copy())
            old_mean=float(source[...,0].mean())
            def encode(region=None,fill="median"):
                changed=data["pooled"].copy()
                changed[0]-=10*(float(inspect(region,fill)["image"][...,0].mean())-old_mean)/255
                return changed
        else:
            xy=d.e.native_xy(data,valid.shape,np.zeros(3),field)
            cols=np.clip(np.rint(xy[:,0]).astype(int),0,63);rows=np.clip(np.rint(xy[:,1]).astype(int),0,63)
            source=rgb[rows,cols].astype(np.float32)/255
            def inspect(region=None,fill="median"):
                color=source.copy()
                if region is not None:color[region]=np.median(source,axis=0) if fill=="median" else 127/255
                return dict(coord=data["meta"]["xyz_scaled"],color=color,normal=np.tile([0,0,1],(len(color),1)).astype(np.float32),grid_origin=np.zeros(3,dtype=np.float32))
            old_mean=float(source[:,0].mean())
            def encode(region=None,fill="median"):
                changed=data["pooled"].copy();changed[0]-=10*(float(inspect(region,fill)["color"][:,0].mean())-old_mean)
                return changed
        encode.inspect_input=inspect
        return encode

    def test_size_and_rim_matching(self):
        valid=np.ones((64,64),bool);xy=d.pixel_xy(valid.shape)
        masks,count=d.matched_masks(xy,valid.ravel(),self.pair,valid.shape,16,min_count=8)
        random,rim=d.random_masks(xy,valid.ravel(),valid,self.pair,masks,count,16,3,123,.5)
        self.assertEqual(masks["plaque"].sum(),masks["clean"].sum())
        self.assertEqual(len(random),3)
        for mask in random:
            self.assertEqual(mask.sum(),count)
            self.assertFalse((mask&(masks["plaque"]|masks["clean"])).any())
            self.assertLessEqual(abs(rim[mask].mean()-rim[masks["plaque"]].mean()),.5)

    def test_overlap_rejected(self):
        pair=dict(self.pair,clean_xy=self.pair["plaque_xy"])
        with self.assertRaisesRegex(ValueError,"overlap"):d.annotation(self.dataset,self.row,[pair])

    def test_reencoder_runs_for_both_fill_styles_and_controls(self):
        outputs=[]
        for encoder in ("dinov3","concerto"):
            args=self.args(encoder)
            out=self.root/("experiment_"+encoder)
            with patch.object(d.e,"make_reencoder",side_effect=self.mock_factory),contextlib.redirect_stdout(io.StringIO()):
                d.run_experiment(args,self.selection,out)
            with (out/"region_effects.csv").open(encoding="utf-8-sig") as stream:
                rows=list(d.csv.DictReader(stream))
            self.assertEqual(len(rows),8)
            for fill in args.fills:
                subset=[r for r in rows if r["fill"]==fill]
                self.assertEqual(len({r["native_edit_count"] for r in subset}),1)
            for r in rows:
                self.assertAlmostEqual(float(r["mean_branch_margin_drop"])+float(r["max_branch_margin_drop"]),float(r["margin_drop"]),places=5)
            with (out/"paired_effects.csv").open(encoding="utf-8-sig") as stream:
                paired=list(d.csv.DictReader(stream))
            self.assertEqual(len(paired),2)
            self.assertTrue(all(np.isfinite(float(r["plaque_minus_clean"])) for r in paired))
            self.assertTrue((out/"pair_01"/"pair_01_median.png").is_file())
            suffix=".png" if encoder=="dinov3" else ".npz"
            self.assertTrue((out/"pair_01"/("plaque_median_encoder_input"+suffix)).is_file())
            if encoder=="concerto":self.assertTrue((out/"pair_01"/"plaque_median_encoder_input.ply").is_file())
            outputs.append(out)
        summary=self.root/"summary"
        d.summarize(SimpleNamespace(results=outputs,out=summary))
        with (summary/"by_specimen.csv").open(encoding="utf-8-sig") as stream:
            rows=list(d.csv.DictReader(stream))
        self.assertEqual(len(rows),4)  # two modes x two fills, each one physical specimen
        self.assertTrue(all(r["physical_id"]=="val" for r in rows))
        with (summary/"summary.csv").open(encoding="utf-8-sig") as stream:
            summaries=list(d.csv.DictReader(stream))
        self.assertTrue(all(r["bootstrap_low"]=="" for r in summaries))

    def test_modified_render_rejected_before_encoder_load(self):
        selection=d.e.read_json(self.selection);selection["rgb_array_sha256"]="wrong"
        file=self.root/"bad_selection.json";d.write_json(file,selection)
        with self.assertRaisesRegex(ValueError,"no longer matches"),patch.object(d.e,"make_reencoder") as factory:
            d.run_experiment(self.args(),file,self.root/"bad_output")
        factory.assert_not_called()

    def test_baseline_mismatch_stops_before_output(self):
        def bad_factory(args,row,data,mode):
            def encode(region=None,fill="median"):return data["pooled"]+100
            return encode
        with patch.object(d.e,"make_reencoder",side_effect=bad_factory),self.assertRaisesRegex(ValueError,"does not reproduce"):
            d.run_experiment(self.args(),self.selection,self.root/"bad_baseline")
        self.assertFalse((self.root/"bad_baseline").exists())

    def test_region_picker_can_save_replayable_selection(self):
        args=SimpleNamespace(run=self.dataset,radius_mm=1.,out=self.root/"picker.json")
        picker=d.RegionPicker(args,self.row)
        for xy in (self.pair["plaque_xy"],self.pair["clean_xy"]):
            picker.on_click(SimpleNamespace(inaxes=picker.axes[0],xdata=xy[0],ydata=xy[1],button=1))
        picker.save();self.assertTrue(args.out.is_file())
        record=d.e.read_json(args.out)
        self.assertEqual(record["pairs"][0]["plaque_xy"],self.pair["plaque_xy"])
        d.plt.close("all")

    def test_summary_averages_pairs_then_scans_then_specimens(self):
        source=self.root/"aggregate_fixture";source.mkdir()
        values=[("A","A1",1,1.),("A","A1",2,3.),("A","A2",1,6.),("B","B1",1,2.),("C","C1",1,3.)]
        records=[]
        for pid,scan,pair,value in values:
            records.append(dict(selection_sha256=scan,physical_id=pid,scan_id=scan,pair_id=pair,
                mode="dinov3_rgb",target="partial",against="clean",intervention="appearance",fill="median",
                plaque_minus_clean=value,plaque_minus_random=value,
                mean_allocation_plaque_minus_clean=value/10,max_allocation_plaque_minus_clean=value/20,
                mean_branch_plaque_minus_clean=value*.4,max_branch_plaque_minus_clean=value*.6))
        d.write_csv(source/"paired_effects.csv",records)
        out=self.root/"aggregate_summary"
        d.summarize(SimpleNamespace(results=[source],out=out))
        with (out/"by_specimen.csv").open(encoding="utf-8-sig") as f:by_pid={r["physical_id"]:r for r in d.csv.DictReader(f)}
        self.assertAlmostEqual(float(by_pid["A"]["plaque_minus_clean"]),4.)
        with (out/"summary.csv").open(encoding="utf-8-sig") as f:s=list(d.csv.DictReader(f))
        self.assertTrue(all(r["specimen_count"]=="3" for r in s))
        self.assertTrue(all(abs(float(r["mean"])-3)<1e-6 for r in s))
        self.assertTrue(all(r["bootstrap_low"]!="" for r in s))


if __name__=="__main__":unittest.main()
