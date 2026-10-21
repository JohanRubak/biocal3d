"""Regenerate the manuscript figure from the included, unchanged TRIOS5 PNGs."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
W, H = 7.2, 7.45
INK = '#172832'; MUTED = '#56656f'; RULE = '#bdc8ce'; ACCENT = '#36656e'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})
fig = plt.figure(figsize=(W, H), facecolor='white')
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')


def text(x, y, label, size=8, bold=False, color=INK, ha='left', va='top'):
    ax.text(x, y, label, fontsize=size, fontweight='bold' if bold else 'normal',
            color=color, ha=ha, va=va, linespacing=1.35)


def heading(x, y, letter, title):
    text(x, y, letter, 11, True)
    text(x + .28, y + .015, title, 9, True)


def line(a, b, color=RULE, lw=.7): ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw)


def arrow(a, b, color=MUTED, lw=.8):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=8, linewidth=lw, color=color))


def image(path, x, y, w, h=None, crop=None):
    im = Image.open(path).convert('RGB')
    if crop is not None: im = im.crop(crop)
    if h is None: h = w * im.height / im.width
    child = fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
    child.imshow(im, interpolation='none'); child.axis('off')
    return im


def common_crop(names):
    bounds = []
    for name in names:
        a = np.asarray(Image.open(ASSETS / name).convert('RGB'))
        yy, xx = np.where(a.min(axis=2) < 245)
        bounds.append([xx.min(), yy.min(), xx.max() + 1, yy.max() + 1])
    bounds = np.asarray(bounds)
    lo = bounds[:, :2].min(axis=0); hi = bounds[:, 2:].max(axis=0)
    center = (lo + hi) / 2; half = max(hi - lo) / 2 + 45
    return tuple(int(round(v)) for v in [center[0]-half, center[1]-half, center[0]+half, center[1]+half])


fresh_crop = common_crop(['fresh_baseline.png', 'fresh_final.png'])
mature_crop = common_crop(['mature_baseline.png', 'mature_partial.png', 'mature_final.png'])

# A: the actual surrounding model and its extracted specimen surface.
heading(.25, .18, 'A', 'ROI extraction from the full scan')
full = np.asarray(Image.open(ASSETS / 'mature_full_partial.png').convert('RGB'))
yy, xx = np.where(full.min(axis=2) < 245)
full_crop = (int(xx.min()-32), int(yy.min()-32), int(xx.max()+33), int(yy.max()+33))
image(ASSETS / 'mature_full_partial.png', .36, .71, 3.55, crop=full_crop)
text(2.1, .54, 'Full TRIOS5 model', 7.8, ha='center')
text(2.1, 1.96, 'Mature example BCG6-4, after partial cleaning', 7.1, color=MUTED, ha='center')
arrow((4.03, 1.23), (4.73, 1.23))
text(4.39, .80, 'ROI\nextraction', 7.1, color=MUTED, ha='center')
image(ASSETS / 'mature_partial.png', 4.91, .64, 1.17, crop=mature_crop)
text(5.50, 1.96, 'Approved specimen surface', 7.1, color=MUTED, ha='center')
line((.25, 2.12), (6.95, 2.12))

# B: each row is one physical specimen followed across its cleaning stages.
heading(.25, 2.30, 'B', 'Cultivation and cleaning stages')
x0, x1, x2 = 2.40, 4.35, 6.28
diameter = 1.05
for label, name, x, stage in [
    ('Baseline (T0)', 'fresh_baseline.png', x0, 'fresh'),
    ('Final cleaning (T1)', 'fresh_final.png', x2, 'fresh'),
    ('Baseline (T0)', 'mature_baseline.png', x0, 'mature'),
    ('Partial cleaning (T1)', 'mature_partial.png', x1, 'mature'),
    ('Final cleaning (T2)', 'mature_final.png', x2, 'mature')]:
    y = 2.91 if stage == 'fresh' else 4.34
    text(x, y - .24, label, 7.6, ha='center')
    image(ASSETS / name, x - diameter / 2, y, diameter, crop=fresh_crop if stage=='fresh' else mature_crop)

text(.30, 2.98, 'Fresh biofilm', 9, True)
text(.30, 3.25, '12 h cultivation\nG1–G2, 16 specimens', 7.6)
text(.30, 3.76, 'Example: BCG1-2', 7, color=MUTED)
arrow((3.05, 3.46), (5.59, 3.46))
text(4.33, 3.18, 'Cleaning', 7.2, color=MUTED, ha='center')
text(.30, 4.42, 'Mature biofilm', 9, True)
text(.30, 4.69, '24 h cultivation\nG3–G6, 32 specimens', 7.6)
text(.30, 5.20, 'Example: BCG6-4', 7, color=MUTED)
arrow((3.04, 4.87), (3.70, 4.87))
arrow((5.00, 4.87), (5.64, 4.87))
line((.25, 5.62), (6.95, 5.62))

# C: every specimen-stage observation is acquired with four systems.
heading(.25, 5.83, 'C', 'Paired scanning')
text(1.77, 6.17, 'Same specimen at each stage', 7.6, ha='center')
arrow((1.77, 6.38), (1.77, 6.54))
line((.63, 6.54), (2.92, 6.54), color=MUTED, lw=.7)
for x, label in zip([.63, 1.39, 2.16, 2.92], ['iTERO', 'TRIOS3', 'TRIOS5', 'LABscanner']):
    arrow((x, 6.54), (x, 6.71), lw=.65)
    text(x, 6.78, label, 6.9, ha='center')

# D: scan count includes all six groups; scans are repeated observations.
heading(3.55, 5.83, 'D', 'Scan structure')
cols = [3.62, 5.36, 6.65]
text(cols[0], 6.18, 'Cohort', 7.2, True)
text(cols[1], 6.18, 'Scans per specimen', 7.2, True, ha='center')
text(cols[2], 6.18, 'Total', 7.2, True, ha='center')
line((3.62, 6.40), (6.95, 6.40))
for y, name, formula, total in [(6.49, 'Fresh', '2 stages × 4 = 8', '128'),
                                 (6.75, 'Mature', '3 stages × 4 = 12', '384')]:
    text(cols[0], y, name, 7.3)
    text(cols[1], y, formula, 7.3, ha='center')
    text(cols[2], y, total, 7.3, ha='center')

text(1.77, 7.06, '4 scans per specimen at each stage', 7.2, ha='center')
line((3.62, 6.99), (6.95, 6.99))
text(5.28, 7.06, '48 specimens, 512 scans', 7.8, True, ha='center')

fig.savefig(ROOT / 'Figure_2_Experimental_Overview.png', dpi=600, facecolor='white')
fig.savefig(ROOT / 'Figure_2_Experimental_Overview.pdf', facecolor='white')
fig.savefig(ROOT / 'Figure_2_Experimental_Overview.svg', facecolor='white')
(ROOT / 'crop_record.json').write_text(json.dumps({'fresh_roi_crop': fresh_crop, 'mature_roi_crop': mature_crop, 'panel_A_full_model_crop': full_crop, 'note': 'ROI stages within each example share an identical crop. A and B have different magnifications.'}, indent=2))
print('Created TRIOS5 PNG, PDF and SVG figure.')
