"""Write an offline threshold viewer for one illustrated scan."""
from base64 import b64encode
from html import escape
from io import BytesIO
import json

import numpy as np
from PIL import Image

from plaque_masks import THRESHOLD
from study_data import STAGES
from study_protocol import INPUTS

STAGE_LABELS = dict(baseline="Baseline", partial="Partial", clean="Fully cleaned")
THRESHOLD_STEPS = np.linspace(0, 1, 101, dtype=np.float32)


def scan_image(rgb, roi):
    image = BytesIO()
    Image.fromarray(np.where(roi[..., None], rgb, 255)).save(image, format="PNG")
    return "data:image/png;base64," + b64encode(image.getvalue()).decode("ascii")


def write_mask_viewer(path, row, raw, roi, scores, predictions, *, harmonized_rgb=None):
    height, width = roi.shape
    title = f"{row['sample_name']} · {row['scanner']}"
    default = round(100 * THRESHOLD)
    cards, score_data = [], []
    for mode, values in scores.items():
        probabilities = predictions[mode]
        guess = STAGES[int(probabilities.argmax())]
        # Bin at the slider's exact cutoffs. Zero marks pixels outside mask support.
        bins = np.searchsorted(THRESHOLD_STEPS, values, side="right")
        encoded = np.where(roi & np.isfinite(values), bins, 0).astype(np.uint8)
        score_data.append(b64encode(encoded.tobytes()).decode("ascii"))
        status = ' data-clean="true"' if guess == "clean" else ""
        background = (' class="harmonized"' if harmonized_rgb is not None and INPUTS[mode][1] == "harmonized"
                      else "")
        name = escape(INPUTS[mode][2])
        cards.append(
            f'<article><h2>{name}</h2>'
            f'<p>Predicted {STAGE_LABELS[guess].lower()} · {probabilities.max():.0%}</p>'
            f'<canvas{background} width="{width}" height="{height}" role="img" '
            f'aria-label="{name} plaque mask"></canvas><p class="coverage"{status}></p></article>')
    data = dict(image=scan_image(raw, roi), scores=score_data, area=int(roi.sum()), default=default)
    preview = ""
    if harmonized_rgb is not None:
        data["harmonized_image"] = scan_image(harmonized_rgb, roi)
        preview = ('<h2>Harmonised scan</h2>'
                   f'<img id="harmonized" width="{width}" height="{height}" alt="Harmonised scan">')
    path.write_text(
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>Plaque threshold · {escape(title)}</title>{STYLE}</head><body>'
        f'<a href="../index.html">Back to study report</a><h1>{escape(title)}</h1>'
        f'<p>Observed stage · {STAGE_LABELS[row["stage"]]}</p>'
        f'<div class="control"><label for="threshold">Plaque threshold</label>'
        f'<input id="threshold" type="range" min="0" max="100" step="1" value="{default}">'
        f'<output id="cutoff" for="threshold">{THRESHOLD:.2f}</output>'
        f'<button id="reset" type="button">Reset</button></div>'
        '<p>Cyan marks pixels at or above the cutoff after CRF refinement. '
        'The same cutoff applies to all models. Fully cleaned predictions keep an empty mask.</p>'
        '<div class="comparison"><article><h2>Original scan</h2>'
        f'<img id="original" width="{width}" height="{height}" alt="Original scan">{preview}</article>'
        f'<section class="models">{"".join(cards)}</section></div>'
        f'<script type="application/json" id="data">{json.dumps(data)}</script>{SCRIPT}</body></html>',
        encoding="utf-8",
    )


STYLE = """<style>
body {font: 16px system-ui, sans-serif; margin: 2rem auto; padding: 0 1rem; max-width: 1120px; color: #183448;}
h1 {margin-bottom: .35rem;}
h2 {font-size: 1rem; margin: 0 0 .4rem;}
article {border: 1px solid #dce5eb; padding: .75rem; border-radius: .4rem; min-width: 0;}
article p {margin: .35rem 0; font-size: .9rem;}
.control {display: flex; align-items: center; gap: .65rem; flex-wrap: wrap;}
input {width: min(440px, 100%); accent-color: #245d84;}
output {font-variant-numeric: tabular-nums; min-width: 3ch;}
button {font: inherit;}
.comparison, .models {display: grid; gap: .75rem; align-items: start;}
.comparison {grid-template-columns: minmax(0, 1fr) minmax(0, 2fr);}
.models {grid-template-columns: repeat(2, minmax(0, 1fr));}
img, canvas {display: block; width: 100%; height: auto;}
canvas {background: var(--scan-image) center / 100% 100% no-repeat;}
canvas.harmonized {background-image: var(--harmonized-image);}
#harmonized {margin-bottom: .35rem;}
@media (max-width: 650px) {.comparison {grid-template-columns: 1fr;}}
</style>"""


SCRIPT = """<script>
const data = JSON.parse(document.getElementById("data").textContent);
document.getElementById("original").src = data.image;
document.body.style.setProperty("--scan-image", 'url("' + data.image + '")');
if (data.harmonized_image) {
    document.getElementById("harmonized").src = data.harmonized_image;
    document.body.style.setProperty("--harmonized-image", 'url("' + data.harmonized_image + '")');
}
const slider = document.getElementById("threshold");
const cutoff = document.getElementById("cutoff");
const statuses = document.querySelectorAll(".coverage");
const panels = Array.from(document.querySelectorAll("canvas"), (canvas, i) => {
    const context = canvas.getContext("2d");
    const overlay = context.createImageData(canvas.width, canvas.height);
    const scores = Uint8Array.from(atob(data.scores[i]), value => value.charCodeAt(0));
    for (let p = 0; p < scores.length; p++) {
        overlay.data[p * 4 + 1] = 199;
        overlay.data[p * 4 + 2] = 230;
    }
    return {context, overlay, scores, status: statuses[i]};
});
function update() {
    const value = Number(slider.value);
    cutoff.value = (value / 100).toFixed(2);
    slider.setAttribute("aria-valuetext", cutoff.value);
    for (const {context, overlay, scores, status} of panels) {
        let count = 0;
        for (let p = 0; p < scores.length; p++) {
            const selected = scores[p] >= value + 1;
            overlay.data[p * 4 + 3] = selected ? 158 : 0;
            count += Number(selected);
        }
        context.putImageData(overlay, 0, 0);
        status.textContent = status.dataset.clean
            ? "Fully cleaned prediction · empty mask"
            : (100 * count / data.area).toFixed(1) + "% of scan ROI";
    }
}
slider.addEventListener("input", update);
document.getElementById("reset").addEventListener("click", () => {
    slider.value = data.default;
    update();
});
update();
</script>"""
