"""Write portable, offline threshold viewers for the selected scans."""

import base64
from html import escape
from io import BytesIO
import json

import numpy as np
from PIL import Image


def write_viewer(path, scan_id, raw, scores, stage):
    image = BytesIO()
    Image.fromarray(raw).save(image, format="PNG")
    payload = json.dumps({
        "scan": scan_id,
        "width": raw.shape[1],
        "height": raw.shape[0],
        "image": base64.b64encode(image.getvalue()).decode("ascii"),
        "scores": {key: base64.b64encode(np.ascontiguousarray(value, dtype="<f4").tobytes()).decode("ascii")
                   for key, value in scores.items()},
    }, separators=(",", ":")).replace("<", "\\u003c")
    page = VIEWER.replace("SCAN_TITLE", escape(scan_id)).replace("SCAN_STAGE", escape(stage))
    path.write_text(page.replace("VIEWER_PAYLOAD", payload), encoding="utf-8")


def write_index(path, panels):
    rows = "".join(
        f'<li><a href="{escape(stem, quote=True)}.html">{escape(scan)}</a> '
        f'<span>{escape(reason)}</span> · <a href="{escape(stem, quote=True)}.png">Static panel</a></li>'
        for stem, scan, reason in panels
    )
    path.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" '
        'content="width=device-width,initial-scale=1"><title>BioCal3D examples</title>'
        '<style>body{font:16px system-ui;color:#172c35;background:#f6f8f9;'
        'max-width:850px;margin:40px auto;padding:0 20px}li{margin:15px 0}a{color:#075f65}'
        'span{color:#53646d}</style><h1>BioCal3D test examples</h1>'
        '<p>Open a scan to threshold the DINOv3 or Concerto mean-pooling map.</p>'
        f'<ul>{rows}</ul></html>',
        encoding="utf-8",
    )


VIEWER = r"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SCAN_TITLE · BioCal3D threshold viewer</title>
<style>
  body { font: 16px system-ui,sans-serif; color:#172c35; background:#f6f8f9; margin:0; }
  main { max-width:1060px; margin:28px auto; padding:0 18px; }
  h1 { margin:.3em 0; } p { margin:.6em 0; }
  a { color:#075f65; }
  .controls,.panel { background:white; border:1px solid #d8e2e5; border-radius:12px; padding:15px; }
  .controls { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px 16px; margin:18px 0; }
  .controls label { display:grid; gap:6px; }
  .controls .slider { grid-column:1/-1; }
  select,button,input { font:inherit; }
  select,input[type=number],button { padding:6px 9px; }
  input[type=range] { width:100%; accent-color:#087c7b; }
  .legend { display:flex; flex-wrap:wrap; gap:10px 18px; margin-bottom:12px; }
  .legend b { display:inline-block; width:13px; height:13px; border-radius:3px; margin-right:6px; }
  .panels { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }
  h2 { font-size:17px; margin:0 0 10px; }
  canvas { width:100%; height:auto; display:block; background:#808080; }
  .small { color:#53646d; font-size:.9rem; }
  @media(max-width:650px) { .panels { grid-template-columns:1fr; } }
</style>
<main>
  <nav><a href="index.html">All examples</a> · <a id="staticPanel">Static panel</a></nav>
  <h1>SCAN_TITLE</h1>
  <p>Recorded stage <strong>SCAN_STAGE</strong></p>
  <p class="small">The mean-pooling score contrasts the average baseline and partial stage logits with the clean logit.
    The threshold turns this model evidence into a candidate mask.</p>
  <section class="controls" aria-label="Threshold controls">
    <label>Model
      <select id="family"><option value="dinov3">DINOv3</option><option value="concerto">Concerto</option></select>
    </label>
    <label class="slider">Plaque score threshold
      <input id="slider" type="range" aria-label="Plaque score threshold">
    </label>
    <label>Threshold in logit units <input id="threshold" type="number" step="any" value="0"></label>
    <label>Overlay
      <span><input id="below" type="checkbox" checked> Show below-threshold region</span>
    </label>
  </section>
  <div class="legend">
    <span><b style="background:#ffd740"></b>Candidate plaque</span>
    <span><b style="background:#36b8ae"></b>Candidate clean (at or below threshold)</span>
    <span class="small">PNG mask has white plaque, black below-threshold pixels and transparent unscored pixels.</span>
  </div>
  <section class="panels">
    <article class="panel"><h2 id="leftTitle">No colour</h2><canvas id="left"></canvas>
      <p class="small" id="leftCount"></p><button id="leftDownload">Download mask PNG</button></article>
    <article class="panel"><h2>Raw colour</h2><canvas id="right"></canvas>
      <p class="small" id="rightCount"></p><button id="rightDownload">Download mask PNG</button></article>
  </section>
</main>
<script type="application/json" id="viewer-data">VIEWER_PAYLOAD</script>
<script>
  "use strict";
  const data = JSON.parse(document.getElementById("viewer-data").textContent);
  const element = (id) => document.getElementById(id);
  const controls = Object.fromEntries(["family","slider","threshold","below"]
    .map((id) => [id, element(id)]));
  const base = new Image();
  const decoded = new Map();
  const plaqueColour = [255,215,64,170];
  const belowColour = [54,184,174,83];
  const panels = ["left", "right"].map((id) => {
    const canvas = element(id);
    const overlay = document.createElement("canvas");
    canvas.width = overlay.width = data.width;
    canvas.height = overlay.height = data.height;
    return {id, canvas, overlay, context: canvas.getContext("2d"), overlayContext: overlay.getContext("2d")};
  });
  let threshold = 0;
  function decode(key) {
    if (!decoded.has(key)) {
      const encoded = atob(data.scores[key]);
      const bytes = Uint8Array.from(encoded, (char) => char.charCodeAt(0));
      const view = new DataView(bytes.buffer);
      const values = new Float32Array(bytes.byteLength / 4);
      for (let i = 0; i < values.length; i++) values[i] = view.getFloat32(i * 4, true);
      decoded.set(key, values);
    }
    return decoded.get(key);
  }
  function setThreshold(value) {
    if (!Number.isFinite(value)) return;
    threshold = value;
    controls.slider.value = String(value);
    controls.threshold.value = String(value);
    paint();
  }
  function updateScores() {
    const family = controls.family.value;
    const modes = family === "dinov3" ? ["dinov3_gray", "dinov3_rgb"] : ["concerto_geometry", "concerto_rgb"];
    element("leftTitle").textContent = family === "dinov3" ? "Grayscale input" : "Geometry input";
    let limit = 1e-8;
    panels.forEach((panel, i) => {
      panel.mode = modes[i];
      panel.scores = decode(modes[i]);
      for (const value of panel.scores) {
        if (Number.isFinite(value)) limit = Math.max(limit, Math.abs(value));
      }
    });
    controls.slider.min = String(-limit);
    controls.slider.max = String(limit);
    controls.slider.step = String(limit / 1000);
    setThreshold(0);
  }
  function paint() {
    for (const panel of panels) {
      if (!panel.scores || !base.complete || !base.naturalWidth) continue;
      const overlay = panel.overlayContext.createImageData(data.width, data.height);
      let scored = 0, plaque = 0;
      for (let i = 0; i < panel.scores.length; i++) {
        const value = panel.scores[i];
        if (!Number.isFinite(value)) continue;
        scored++;
        const candidate = value > threshold;
        if (candidate) plaque++;
        if (!candidate && !controls.below.checked) continue;
        overlay.data.set(candidate ? plaqueColour : belowColour, 4 * i);
      }
      panel.overlayContext.putImageData(overlay, 0, 0);
      panel.context.drawImage(base, 0, 0);
      panel.context.drawImage(panel.overlay, 0, 0);
      const fraction = scored ? (100 * plaque / scored).toFixed(1) + "%" : "—";
      element(panel.id + "Count").textContent = fraction + " candidate plaque among scored ROI pixels";
    }
  }
  function download(panel) {
    const mask = document.createElement("canvas");
    mask.width = data.width;
    mask.height = data.height;
    const context = mask.getContext("2d");
    const image = context.createImageData(mask.width, mask.height);
    for (let i = 0; i < panel.scores.length; i++) {
      const value = panel.scores[i];
      if (!Number.isFinite(value)) continue;
      const pixel = value > threshold ? 255 : 0;
      image.data.set([pixel,pixel,pixel,255], 4 * i);
    }
    context.putImageData(image, 0, 0);
    const file = [data.scan, panel.mode, "mean", "threshold", threshold].join("_");
    mask.toBlob((blob) => {
      if (!blob) return;
      const link = document.createElement("a");
      link.download = file.replace(/[^A-Za-z0-9_.-]/g, "_") + ".png";
      link.href = URL.createObjectURL(blob);
      document.body.append(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    });
  }
  element("staticPanel").href = location.pathname.split("/").at(-1).replace(/\.html$/, ".png");
  controls.family.addEventListener("change", updateScores);
  controls.slider.addEventListener("input", () => setThreshold(controls.slider.valueAsNumber));
  controls.threshold.addEventListener("change", () => setThreshold(controls.threshold.valueAsNumber));
  controls.below.addEventListener("change", paint);
  for (const panel of panels) element(panel.id + "Download").addEventListener("click", () => download(panel));
  base.onload = paint;
  base.src = "data:image/png;base64," + data.image;
  updateScores();
</script>
</html>
"""
