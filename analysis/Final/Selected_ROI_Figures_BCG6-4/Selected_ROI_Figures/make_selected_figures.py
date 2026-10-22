"""Reproduce selected figures from the supplied ROI exploration report/cache.

Usage: python make_selected_figures.py REPORT_DIRECTORY [OUTPUT_DIRECTORY]
Requires numpy, pandas, matplotlib, scipy, pillow. No original meshes needed.
Neutral views are height-grid renderings, not native-mesh renderings.
"""
from pathlib import Path
from io import BytesIO
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('Selected_ROI_Figures')
OUT.mkdir(parents=True, exist_ok=True)
S = ['iTERO', 'TRIOS3', 'TRIOS5', 'LABscanner']
C = dict(zip(S, ['#0072B2', '#E69F00', '#009E73', '#AA4499']))
STAGES = ['baseline', 'partial', 'clean']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.titlesize': 10, 'axes.labelsize': 9,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none',
                     'axes.spines.top': False, 'axes.spines.right': False})
scan = pd.read_csv(ROOT/'tables/scan_measurements.csv')
paired = pd.read_csv(ROOT/'tables/paired_stage_changes.csv')
scanner_summary = pd.read_csv(ROOT/'tables/paired_scanner_summary.csv')
stage_summary = pd.read_csv(ROOT/'tables/paired_stage_summary.csv')


def save(fig, name):
    for extension in ['png', 'pdf']:
        b = BytesIO()
        fig.savefig(b, format=extension, dpi=300, bbox_inches='tight', facecolor='white')
        dest = OUT/f'{name}.{extension}'
        temp = dest.with_suffix(dest.suffix+'.tmp')
        with temp.open('wb') as f:
            f.write(b.getvalue()); f.flush(); os.fsync(f.fileno())
        os.replace(temp, dest)
    plt.close(fig)


def clean(ax):
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)


# Colour summary: paired scanners at baseline and paired specimen cleaning response.
fig = plt.figure(figsize=(7.5, 6.7), layout='constrained')
gs = fig.add_gridspec(2, 2, height_ratios=[1, .95])
for j, cohort in enumerate(['fresh', 'mature']):
    ax = fig.add_subplot(gs[0, j])
    a = np.zeros((4, 4))
    d = scanner_summary.query("cohort == @cohort and stage == 'baseline' and metric == 'deltaE00_mean_Lab'")
    for row in d.itertuples():
        i, k = S.index(row.scanner_a), S.index(row.scanner_b)
        a[i, k] = a[k, i] = row.median
    ax.imshow(a, cmap='magma', vmin=0, vmax=16)
    for i in range(4):
        for k in range(4):
            ax.text(k, i, '—' if i == k else f'{a[i,k]:.1f}', ha='center', va='center',
                    color='black' if a[i,k] > 10 else 'white', fontsize=10)
    ax.set(xticks=range(4), yticks=range(4), xticklabels=S, yticklabels=S,
           title=f'{chr(65+j)}  {cohort.capitalize()} baseline (n = {16 if j == 0 else 32})')
    ax.tick_params(length=0, axis='x', labelrotation=25)
    ax.set_xlabel('Median paired ΔE₀₀ between scan-mean colours', fontsize=8)
ax = fig.add_subplot(gs[1, :])
for j, cohort in enumerate(['fresh', 'mature']):
    for i, scanner in enumerate(S):
        values = paired.query("cohort == @cohort and scanner == @scanner and transition == 'baseline_to_clean'").deltaE00_mean_Lab.to_numpy()
        x = i + (-.16 if j == 0 else .16)
        jitter = np.linspace(-.085, .085, len(values))
        ax.scatter(x+jitter, values, s=10, color=C[scanner], alpha=.26, linewidths=0)
        row = stage_summary.query("cohort == @cohort and scanner == @scanner and transition == 'baseline_to_clean' and metric == 'deltaE00_mean_Lab'").iloc[0]
        ax.errorbar(x, row['median'], yerr=[[row['median']-row.ci_low], [row.ci_high-row['median']]],
                    fmt='o' if j == 0 else 's', ms=6, color=C[scanner],
                    mfc='white' if j == 0 else C[scanner], capsize=3, lw=1.4)
        ax.annotate(f"{row['median']:.1f}", (x, row.ci_high), xytext=(0, 5),
                    textcoords='offset points', ha='center', fontsize=8)
ax.plot([], [], 'o', color='#555555', mfc='white', label='Fresh (n = 16)')
ax.plot([], [], 's', color='#555555', label='Mature (n = 32)')
ax.set(xticks=range(4), xticklabels=S, ylabel='Baseline-to-clean ΔE₀₀',
       title='C  Colour change after cleaning: paired within specimens', ylim=(0, 20))
ax.legend(frameon=False, ncol=2, loc='upper right', fontsize=8)
clean(ax)
save(fig, 'Main_1_Colour_differences')


# Geometric summary: sampling, local slopes, and plane-detrended height amplitude.
fig, axes = plt.subplots(2, 3, figsize=(9.2, 6.2), layout='constrained')
metrics = [('faces_per_mm2', 'Triangles per mm²', True),
           ('slope_median_deg', 'Median native face slope (°)', False),
           ('grid_100um_height_rms_um', 'RMS height at 100 µm cells (µm)', True)]
for r, cohort in enumerate(['fresh', 'mature']):
    stages = ['baseline', 'clean'] if cohort == 'fresh' else STAGES
    for k, (metric, label, logarithmic) in enumerate(metrics):
        ax = axes[r, k]
        for j, scanner in enumerate(S):
            medians = []
            for i, stage in enumerate(stages):
                values = scan.query('cohort == @cohort and scanner == @scanner and stage == @stage')[metric].to_numpy()
                x = i+(j-1.5)*.055
                ax.scatter(x+np.linspace(-.017,.017,len(values)), values,
                           s=9, color=C[scanner], alpha=.19, linewidths=0)
                q = np.quantile(values, [.25,.5,.75]); medians.append(q[1])
                ax.errorbar(x, q[1], yerr=[[q[1]-q[0]],[q[2]-q[1]]],
                            fmt='o', ms=3.5, color=C[scanner], lw=1.1, capsize=2)
            ax.plot(np.arange(len(stages))+(j-1.5)*.055, medians,
                    color=C[scanner], label=scanner, lw=1.1)
        ax.set(xticks=range(len(stages)), xticklabels=[x.capitalize() for x in stages],
               ylabel=label, title=f'{chr(65+3*r+k)}  {cohort.capitalize()} (n = {16 if r == 0 else 32})')
        if logarithmic: ax.set_yscale('log')
        clean(ax)
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='outside lower center', ncol=4, frameon=False)
save(fig, 'Main_2_Mesh_and_topography')


# Use exactly the report's shared height limit, across all 512 saved height maps.
all_heights = []
for file in (ROOT/'cache').glob('*.npz'):
    with np.load(file, allow_pickle=False) as a:
        h = a['view_height']; all_heights.append(np.abs(h[np.isfinite(h)])*1000)
limit = float(np.quantile(np.concatenate(all_heights), .98))
del all_heights


def cached(scanner, t):
    with np.load(ROOT/'cache'/f'{scanner}__BCG6T{t}-4.npz', allow_pickle=False) as a:
        return {k: a[k].copy() for k in a.files if k in ['view_rgb','view_valid','view_height','fov_mm']}


def rgb(a):
    im = a['view_rgb'].copy(); im[~a['view_valid']] = 255
    return im


def surface_shade(height, pitch_mm=.05, exaggeration=20):
    valid = np.isfinite(height)
    # Erode one cell so artificial derivatives at missing-data boundaries are hidden.
    supported = binary_erosion(valid)
    filled = np.where(valid, height, 0)
    dy, dx = np.gradient(filled*exaggeration, pitch_mm, pitch_mm)
    normals = np.dstack([-dx, dy, np.ones_like(height)])
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)
    light = np.array([.4,-.5,1.]); light /= np.linalg.norm(light)
    shade = .25+.75*np.clip(normals@light, 0, 1)
    shade[~supported] = np.nan
    return shade


fig, axes = plt.subplots(3, 4, figsize=(9, 7.2), layout='constrained')
for j, scanner in enumerate(S):
    a = cached(scanner, 1); fov = float(a['fov_mm']); e = [-fov/2,fov/2,-fov/2,fov/2]
    axes[0,j].imshow(rgb(a), extent=e)
    axes[1,j].imshow(surface_shade(a['view_height']), extent=e, cmap='gray', vmin=0,vmax=1)
    image = axes[2,j].imshow(a['view_height']*1000, extent=e, cmap='coolwarm', vmin=-limit,vmax=limit)
    axes[0,j].set_title(scanner)
    for i in range(3):
        ax = axes[i,j]
        ax.set(xticks=[-5,0,5], yticks=[-5,0,5], aspect='equal')
        if i == 2: ax.set_xlabel('x (mm)')
        else: ax.tick_params(labelbottom=False)
        if j: ax.tick_params(labelleft=False)
        else: ax.set_ylabel(['A  Unlit RGB','B  Neutral shading\n20× vertical relief','C  Detrended height'][i]+'\ny (mm)')
fig.colorbar(image, ax=axes[2,:].tolist(), location='bottom', shrink=.55,
             aspect=35, label='Height relative to fitted plane (µm)', extend='both')
fig.suptitle('BCG6-4 · partial cleaning · 13.4 mm field of view', fontsize=12)
save(fig, 'Main_3_BCG6-4_Colour_and_topography')


fig, axes = plt.subplots(3, 4, figsize=(9, 7.1), layout='constrained')
for i, stage in enumerate(STAGES):
    for j, scanner in enumerate(S):
        a = cached(scanner,i); fov = float(a['fov_mm'])
        ax = axes[i,j];ax.imshow(rgb(a), extent=[-fov/2,fov/2,-fov/2,fov/2])
        ax.set(xticks=[-5,0,5], yticks=[-5,0,5], aspect='equal')
        if i == 0: ax.set_title(scanner)
        if j == 0: ax.set_ylabel(stage.capitalize()+'\ny (mm)')
        else: ax.tick_params(labelleft=False)
        if i == 2: ax.set_xlabel('x (mm)')
        else: ax.tick_params(labelbottom=False)
fig.suptitle('BCG6-4 · colour across experimental stages', fontsize=12)
save(fig, 'Supplement_BCG6-4_Stage_series')


# Save exact evidence behind the headline comparisons.
selected = stage_summary.query("transition == 'baseline_to_clean' and metric in ['deltaE00_mean_Lab','delta_L_mean','delta_a_mean','delta_b_mean','delta_grid_100um_height_rms_um','delta_slope_median_deg','face_density_ratio']")
selected.to_csv(OUT/'Key_paired_baseline_to_clean.csv', index=False)
scan.query("physical_id == 'BCG6-4'").to_csv(OUT/'BCG6-4_scan_measurements.csv',index=False)
scanner_summary.query("metric == 'deltaE00_mean_Lab'").to_csv(OUT/'Paired_scanner_colour_summary.csv', index=False)
(OUT/'Figure_settings.json').write_text(json.dumps({
    'source_run': ROOT.name, 'specimen': 'BCG6-4', 'main_example_stage': 'partial',
    'height_limit_um': limit, 'height_cell_width_mm': .05, 'height_colour_clipped': True,
    'relief_exaggeration': 20, 'RGB_source': '320x320 cached unlit preview; no colour correction',
    'neutral_geometry_source': '50 um height grid, NOT native mesh',
    'cross_scanner_pointwise_alignment': False,
    'main_1_error_bars': '95% bootstrap CI of median, from supplied specimen-level summaries',
    'main_2_error_bars': 'IQR, individual scans shown, specimens paired across stages in source tables'
}, indent=2))
print(f'Created four PNG/PDF figures in {OUT.resolve()}; shared height limit ±{limit:.4f} µm.')
