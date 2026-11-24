"""Save paired mask panels and an offline threshold comparison."""

import base64
from html import escape
from io import BytesIO
import json

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image

from stage_ablation import save_npz
from study_data import STAGES

LABELS = {"dinov3_gray": "DINOv3 · grayscale", "dinov3_rgb": "DINOv3 · raw RGB",
          "concerto_geometry": "Concerto · geometry", "concerto_rgb": "Concerto · geometry + RGB"}
MASK_COLOUR = ListedColormap(["#ffd740"])


def write_index(path, panels):
    links = "".join(f'<li><a href="{escape(scan, quote=True)}.html">{escape(scan)}</a> '
                    f'· {escape(stage)} · <a href="{escape(scan, quote=True)}.png">static panel</a></li>'
                    for scan, stage in panels)
    path.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>BioCal3D pilot examples</title>'
        '<style>body{font:16px system-ui;max-width:850px;margin:35px auto;padding:0 16px;'
        'background:#f6f8f9;color:#172c35}li{margin:12px 0}a{color:#075f65}</style>'
        '<h1>BioCal3D pseudo-label comparison</h1>'
        '<p>Open a scan to compare the existing mean-pooling mask with the paper-style attention mask.</p>'
        f'<ul>{links}</ul></html>', encoding="utf-8")


def mask_png(path, score, threshold):
    valid = np.isfinite(score)
    binary = ((score > threshold) * 255).astype(np.uint8)
    Image.fromarray(np.stack((binary, binary, binary, valid.astype(np.uint8) * 255), axis=-1)).save(path)


def export_panel(out, row, raw, data, mean_threshold, paper_threshold):
    scan, stage = row["scan_id"], row["stage"]
    fig, grid = plt.subplots(len(data), 4, figsize=(12, 12), layout="constrained")
    arrays = {"mean_threshold": np.array(mean_threshold), "paper_threshold": np.array(paper_threshold)}
    viewer_scores = {}
    for axes, (mode, item) in zip(grid, data.items()):
        mean, attention, paper = (item[key] for key in ("mean", "attention", "paper"))
        for key, value in item.items():
            arrays[f"{mode}_{key}"] = np.asarray(value)
        viewer_scores[mode] = {key: base64.b64encode(np.ascontiguousarray(
            item[key], dtype="<f4").tobytes()).decode("ascii") for key in ("mean", "paper")}
        mask_png(out / f"{scan}_{mode}_mean_mask.png", mean, mean_threshold)
        mask_png(out / f"{scan}_{mode}_paper_mask.png", paper, paper_threshold)
        for axis in axes:
            axis.imshow(raw)
            axis.set_axis_off()
        finite = np.abs(mean[np.isfinite(mean)])
        limit = max(1e-8, float(np.quantile(finite, .99)))
        axes[0].imshow(np.ma.masked_invalid(mean), cmap="coolwarm", vmin=-limit, vmax=limit, alpha=.78)
        axes[1].imshow(np.ma.masked_invalid(attention), cmap="magma", vmin=0, vmax=1, alpha=.78)
        for axis, score, threshold in ((axes[2], mean, mean_threshold), (axes[3], paper, paper_threshold)):
            mask = score > threshold
            axis.imshow(np.ma.masked_where(~mask, mask), cmap=MASK_COLOUR, vmin=0, vmax=1, alpha=.72)
        stage_guess = item["prediction"]
        probability = item["stage_probabilities"][STAGES.index(stage_guess)]
        gate = "closed" if stage_guess == "clean" else "open"
        headings = (f"{LABELS[mode]}\nMean allocation", "Learned attention",
                    f"Mean-pooling mask · {mean_threshold:g}",
                    f"Paper-style mask · {paper_threshold:g}\n{stage_guess.title()} {probability:.1%} · gate {gate}")
        for axis, label in zip(axes, headings):
            axis.set_title(label, fontsize=9)
    fig.suptitle(f"{scan} · recorded stage {stage}\nCandidate masks compared on the same ROI support", fontsize=12)
    fig.savefig(out / f"{scan}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    save_npz(out / f"{scan}_maps.npz", **arrays)

    buffer = BytesIO()
    Image.fromarray(raw).save(buffer, format="PNG")
    payload = json.dumps(dict(scan=scan, width=raw.shape[1], height=raw.shape[0],
                              image=base64.b64encode(buffer.getvalue()).decode("ascii"),
                              scores=viewer_scores,
                              predictions={mode: dict(stage=item["prediction"],
                                                      probabilities=dict(zip(STAGES, map(float, item["stage_probabilities"]))))
                                           for mode, item in data.items()},
                              mean_threshold=mean_threshold,
                              paper_threshold=paper_threshold)).replace("<", "\\u003c")
    page = VIEWER.replace("SCAN_NAME", escape(scan)).replace("PAYLOAD", payload)
    (out / f"{scan}.html").write_text(page, encoding="utf-8")


VIEWER = r"""<!doctype html><html lang="en">
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SCAN_NAME · Pseudo-label comparison</title>
<style>
body{font:16px system-ui;color:#172c35;background:#f6f8f9;max-width:1050px;margin:24px auto;padding:0 16px}
a{color:#075f65} .grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
article{border:1px solid #d8e2e5;border-radius:12px;padding:14px;background:#fff}
h2{font-size:17px;margin:0 0 8px} canvas{display:block;width:100%;height:auto;background:#808080}
input[type=range]{width:100%;accent-color:#087c7b} label{display:block;margin:10px 0}
select,button{font:inherit;padding:6px} @media(max-width:600px){.grid{grid-template-columns:1fr}}
</style>
<a href="index.html">All examples</a> · <a id="static">Static comparison</a>
<h1>SCAN_NAME</h1>
<label>Model and input <select id="mode"></select></label>
<p id="probability"></p>
<p>Yellow shows candidate plaque. Each method uses its own threshold and score scale.</p>
<section class="grid">
<article><h2>Existing mean-pooling mask</h2><canvas id="meanCanvas"></canvas>
<label>Mean-pooling threshold <input id="meanSlider" type="range"></label>
<p id="meanCount"></p><button id="meanDownload">Download mask PNG</button></article>
<article><h2>Paper-style learned attention mask</h2><canvas id="paperCanvas"></canvas>
<label>Refined score threshold <input id="paperSlider" type="range" min="0" max="1" step=".001"></label>
<p id="paperCount"></p><button id="paperDownload">Download mask PNG</button></article>
</section>
<script id="data" type="application/json">PAYLOAD</script>
<script>
"use strict";
const el = id => document.getElementById(id);
const data = JSON.parse(el("data").textContent);
const labels = {dinov3_gray:"DINOv3 · grayscale",dinov3_rgb:"DINOv3 · raw RGB",
                concerto_geometry:"Concerto · geometry",concerto_rgb:"Concerto · geometry + RGB"};
for (const mode of Object.keys(data.scores)) el("mode").add(new Option(labels[mode], mode));
el("static").href = location.pathname.split("/").pop().replace(/\.html$/, ".png");
const base = new Image();
const panels = {};
for (const key of ["mean", "paper"]) {
  const canvas = el(key+"Canvas");
  const overlay = document.createElement("canvas");
  canvas.width = overlay.width = data.width;
  canvas.height = overlay.height = data.height;
  panels[key] = {canvas, overlay, slider:el(key+"Slider"), values:null};
}
function decode(encoded) {
  const bytes = Uint8Array.from(atob(encoded), char => char.charCodeAt(0));
  const view = new DataView(bytes.buffer);
  const result = new Float32Array(bytes.length / 4);
  for (let i=0;i<result.length;i++) result[i] = view.getFloat32(i*4,true);
  return result;
}
function changeMode() {
  const mode = el("mode").value;
  const scores = data.scores[mode];
  const prediction = data.predictions[mode];
  const probabilities = Object.entries(prediction.probabilities)
    .map(([stage, probability]) => stage + " " + (100*probability).toFixed(1) + "%").join(" · ");
  el("probability").textContent = "Predicted stage " + prediction.stage + " · " + probabilities +
    " · attention mask gate " + (prediction.stage === "clean" ? "closed" : "open");
  for (const key of ["mean", "paper"]) panels[key].values = decode(scores[key]);
  let limit = Math.max(1e-8, Math.abs(data.mean_threshold));
  for (const value of panels.mean.values) if (Number.isFinite(value)) limit = Math.max(limit,Math.abs(value));
  panels.mean.slider.min = -limit;
  panels.mean.slider.max = limit;
  panels.mean.slider.step = "any";
  panels.mean.slider.value = data.mean_threshold;
  panels.paper.slider.value = data.paper_threshold;
  paint();
}
function pixels(key, context, binary) {
  const {values, slider} = panels[key];
  const image = context.createImageData(data.width,data.height);
  const threshold = Number(slider.value);
  let count=0, support=0;
  for (let i=0;i<values.length;i++) {
    const value=values[i];
    if (!Number.isFinite(value)) continue;
    support++;
    const positive=value>threshold;
    if (positive) count++;
    const j=i*4;
    if (binary) {
      image.data[j]=image.data[j+1]=image.data[j+2]=positive?255:0;
      image.data[j+3]=255;
    } else if (positive) {
      image.data[j]=255; image.data[j+1]=215; image.data[j+2]=64; image.data[j+3]=170;
    }
  }
  return {image,count,support};
}
function paint() {
  if (!base.complete || !base.naturalWidth) return;
  for (const key of ["mean","paper"]) {
    const {canvas,overlay,slider} = panels[key];
    const layer = overlay.getContext("2d");
    const result = pixels(key,layer,false);
    layer.putImageData(result.image,0,0);
    const context=canvas.getContext("2d");
    context.drawImage(base,0,0);
    context.drawImage(overlay,0,0);
    el(key+"Count").textContent = "Threshold "+Number(slider.value).toPrecision(5)+" · "+
      (result.support ? (100*result.count/result.support).toFixed(1)+"% of shared ROI" : "no score");
  }
}
for (const key of ["mean","paper"]) {
  panels[key].slider.addEventListener("input",paint);
  el(key+"Download").addEventListener("click",()=>{
    const mask=document.createElement("canvas");
    mask.width=data.width; mask.height=data.height;
    const context=mask.getContext("2d");
    context.putImageData(pixels(key,context,true).image,0,0);
    const filename=[data.scan,el("mode").value,key,panels[key].slider.value].join("_");
    mask.toBlob(blob=>{
      if(!blob)return;
      const link=document.createElement("a");
      link.download=filename.replace(/[^a-zA-Z0-9_.-]/g,"_")+".png";
      link.href=URL.createObjectURL(blob);
      document.body.append(link);link.click();link.remove();
      setTimeout(()=>URL.revokeObjectURL(link.href),1000);
    });
  });
}
el("mode").addEventListener("change",changeMode);
base.onload=paint;
changeMode();
base.src="data:image/png;base64,"+data.image;
</script></html>
"""
