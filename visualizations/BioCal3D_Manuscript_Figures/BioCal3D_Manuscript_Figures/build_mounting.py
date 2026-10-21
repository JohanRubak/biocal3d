"""Label the actual TRIOS5 mounting arrangement without altering image colours."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image

ROOT = Path(__file__).resolve().parent
W, H = 3.5, 2.30
plt.rcParams.update({'font.family':'DejaVu Sans', 'svg.fonttype':'none', 'pdf.fonttype':42})
fig = plt.figure(figsize=(W, H), facecolor='white')
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')
ax.set_zorder(5); ax.patch.set_alpha(0)
crop = (185, 480, 1440, 1060)
x, y, width = .12, .70, 3.26
im = Image.open(ROOT/'assets'/'fresh_full_baseline.png').convert('RGB').crop(crop)
height = width * im.height / im.width
photo = fig.add_axes([x/W, 1-(y+height)/H, width/W, height/H])
photo.imshow(im, interpolation='none'); photo.axis('off')


def point(px, py): return x + (px-crop[0])*width/im.width, y + (py-crop[1])*height/im.height


def label(title, tx, start, target):
    ax.text(tx, .13, title, ha='center', va='top', fontsize=8.5, color='#172832', linespacing=1.2)
    # Straight leaders point to observed anatomical/model features.
    ax.add_patch(FancyArrowPatch(start, point(*target), arrowstyle='-|>', mutation_scale=5.5,
                                linewidth=.65, color='#253d49', shrinkA=0, shrinkB=0))


label('Dental\nmodel', .46, (.46, .55), (440, 770))
label('Specimen\nholder', 1.65, (1.65, .55), (670, 690))
label('Circular\nspecimen', 2.97, (2.97, .55), (855, 815))
for suffix in ('png', 'pdf', 'svg'):
    fig.savefig(ROOT/f'Figure_1_Mounting_Arrangement.{suffix}', dpi=600, facecolor='white')
(ROOT/'mounting_annotation_record.json').write_text(json.dumps({'source':'assets/fresh_full_baseline.png',
 'crop':crop, 'leader_targets_in_source_pixels':{'dental_model':[440,770],'specimen_holder':[670,690],
 'circular_specimen':[855,815]}, 'policy':'No colour correction or reconstruction. Labels and arrows overlaid on the supplied scan rendering.'},indent=2))
print('Created labelled mounting figure, PNG/PDF/SVG.')
