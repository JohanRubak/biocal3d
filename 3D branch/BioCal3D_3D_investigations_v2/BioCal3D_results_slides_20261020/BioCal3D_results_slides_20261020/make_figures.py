"""Reproduce slide figures from the uploaded BioCal3D review export.

Usage: python make_figures.py /path/to/extracted/review
All quantities are observed validation results; no new model is fitted.
"""
from pathlib import Path
import sys, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SOURCE=Path(sys.argv[1]) if len(sys.argv)>1 else Path('../review_20261020')
OUT=Path(__file__).resolve().parent
BLUE='#2878A5'; GREEN='#258776'; ORANGE='#CE7549'; GRAY='#9DA9B3'; DARK='#20384A'
COLORS={'pointnext_s':BLUE,'concerto':GREEN}
NAMES={'pointnext_s':'PointNeXt-S','concerto':'Concerto'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':16,
    'axes.labelsize':12,'xtick.labelsize':11,'ytick.labelsize':12,'axes.spines.top':False,
    'axes.spines.right':False,'axes.labelcolor':DARK,'text.color':DARK,'axes.edgecolor':'#AAB6BF',
    'svg.fonttype':'none','savefig.facecolor':'white'})
allruns=pd.read_csv(SOURCE/'followup_comparison.csv')
summary=pd.read_csv(SOURCE/'colour_pathway_summary.csv')
matched=allruns[allruns.run.str.startswith('reference_followup__')]
tables=OUT/'source_tables';tables.mkdir(exist_ok=True)
summary.to_csv(tables/'colour_pathway_summary.csv',index=False)

def save(fig,name):
    for ext in ['png','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=220)
    plt.close(fig)

def style(ax):
    ax.set_axisbelow(True);ax.grid(axis='x',color='#E6EBEF',linewidth=.8)
    ax.set_xlim(0,110);ax.set_xticks(np.arange(0,101,20));ax.set_xlabel('Validation specimen-balanced accuracy (%)')

def foot(fig,text):fig.text(.07,.035,text,fontsize=10,color='#566977',va='bottom')

# 1. Architecture-matched geometry versus RGB study, with individual seeds.
fig,ax=plt.subplots(figsize=(12,5.7));fig.subplots_adjust(left=.18,right=.94,bottom=.22,top=.8)
for i,model in enumerate(['pointnext_s','concerto']):
    for j,(path,label) in enumerate([('geometry_matched','Geometry only'),('rgb_combined','RGB + geometry')]):
        y=i*1.6+j*.52
        row=summary[(summary.model==model)&(summary.pathway==path)].iloc[0]
        ax.barh(y,row.BA_mean*100,height=.36,color=GRAY if j==0 else COLORS[model],
                xerr=row.BA_seed_SD*100,error_kw={'ecolor':DARK,'capsize':4,'lw':1.3})
        vals=matched[(matched.model==model)&(matched.variant==path)].specimen_balanced_accuracy.to_numpy()*100
        ax.scatter(vals,np.full(3,y),color=DARK,s=17,zorder=4)
        ax.text(103,y,f'{row.BA_mean*100:.1f}%',va='center',fontsize=13,fontweight='bold')
        ax.text(-2,y,label,ha='right',va='center',fontsize=11)
    ax.text(0,i*1.6-.42,NAMES[model],fontsize=14,fontweight='bold')
ax.set_yticks([]);ax.invert_yaxis();style(ax)
fig.suptitle('RGB strongly improves stage classification',x=.07,ha='left',fontsize=22,fontweight='bold',y=.96)
fig.text(.07,.875,'Matched encoder input width and classifier architecture within each model',fontsize=12)
foot(fig,'Mean ± seed SD; dots = seeds 42/43/44. Same 64 validation scans / 6 specimens; 4 partial specimens.\nValidation selects checkpoints. These are not held-out test or scanner-holdout results.')
save(fig,'01_geometry_vs_rgb')

# 2. Entire prespecified six-pathway comparison; no cherry-picked variants.
paths=['geometry_matched','stats_only','rgb_encoder_only','rgb_combined','rgb_shuffled','gray_combined']
labels=['Geometry only','Six global colour statistics','RGB encoder only','RGB encoder + colour statistics','RGB shuffled during training + validation','Grayscale during training + validation']
fig,ax=plt.subplots(figsize=(14,7.7));fig.subplots_adjust(left=.37,right=.91,bottom=.18,top=.81)
for mi,model in enumerate(['pointnext_s','concerto']):
    for i,path in enumerate(paths):
        row=summary[(summary.model==model)&(summary.pathway==path)].iloc[0];y=i+(mi-.5)*.32
        ax.barh(y,row.BA_mean*100,height=.27,color=COLORS[model],label=NAMES[model] if i==0 else None,
            xerr=row.BA_seed_SD*100,error_kw={'ecolor':DARK,'capsize':3,'lw':1})
        ax.text(103,y,f'{row.BA_mean*100:.1f}',va='center',fontsize=11,color=COLORS[model])
ax.set_yticks(range(len(paths)),labels);ax.invert_yaxis();style(ax)
ax.axvline(100/3,color='#99A5AE',linestyle=':',zorder=0)
ax.legend(loc='lower left',bbox_to_anchor=(0,1.025),ncol=2,frameon=False,fontsize=11)
fig.suptitle('Spatial colour arrangement is not essential for high validation scores',x=.055,ha='left',fontsize=19,fontweight='bold',y=.97)
fig.text(.055,.89,'Shuffled RGB retains performance; simple means/maxima explain only part of the signal',fontsize=12)
foot(fig,'Mean ± seed SD across 3 fits on the same validation cohort. Shuffling preserves each scan’s marginal RGB distribution.\nThis does not establish plaque localization or absence of any geometric contribution.')
save(fig,'02_colour_pathways')

# 3. Paired training runs for the interior and separate inference-only rotation probe.
interior=pd.read_csv(SOURCE/'interior_comparison.csv');interior.to_csv(tables/'interior_comparison.csv',index=False)
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,6.2));fig.subplots_adjust(left=.07,right=.97,bottom=.25,top=.77,wspace=.35)
for i,path in enumerate(['geometry_matched','stats_only','rgb_combined']):
    rr=interior[(interior.model=='pointnext_s')&(interior.variant==path)].sort_values('seed')
    for j,(_,r) in enumerate(rr.iterrows()):
        ax1.plot([i-.18,i+.18],[r.full_specimen_BA*100,r.interior_specimen_BA*100],color='#BEC9D1',lw=1.2,zorder=1)
    ax1.scatter(np.full(len(rr),i-.18),rr.full_specimen_BA*100,color=BLUE,s=45,label='Full ROI' if i==0 else None,zorder=2)
    ax1.scatter(np.full(len(rr),i+.18),rr.interior_specimen_BA*100,color=ORANGE,s=45,label='Inner ROI' if i==0 else None,zorder=2)
    ax1.text(i,107,f'{rr.full_specimen_BA.mean()*100:.1f} → {rr.interior_specimen_BA.mean()*100:.1f}',ha='center',fontsize=11)
ax1.set_xticks(range(3),['Geometry','Global colour\nstatistics','RGB +\ngeometry']);ax1.set_ylim(25,113)
ax1.set_yticks(np.arange(30,101,10));ax1.set_ylabel('Validation specimen-balanced accuracy (%)');ax1.set_title('Interior retraining: PointNeXt-S',pad=22)
ax1.grid(axis='y',color='#E6EBEF');ax1.set_axisbelow(True);ax1.legend(loc='lower right',fontsize=10,frameon=False)
sensitivity=pd.read_csv(SOURCE/'input_sensitivity.csv')
rot=sensitivity[(sensitivity.tag=='followup')&(sensitivity.variant=='rgb_combined')&(sensitivity.seed==42)&(sensitivity['transform']=='rotate_z_90')]
rot.to_csv(tables/'rotation_seed42.csv',index=False)
for i,model in enumerate(['pointnext_s','concerto']):
    row=rot[rot.model==model].iloc[0]
    for j,(value,label,color) in enumerate([(row.specimen_balanced_accuracy*100,'Original',BLUE),(row.perturbed_specimen_BA*100,'90° rotation',ORANGE)]):
        x=i+(j-.5)*.32;ax2.bar(x,value,width=.28,color=color,label=label if i==0 else None)
        ax2.text(x,value+1.4,f'{value:.1f}',ha='center',fontsize=11)
ax2.set_xticks([0,1],['PointNeXt-S','Concerto']);ax2.set_ylim(0,113);ax2.set_title('Rotation at inference: seed 42',pad=22)
ax2.grid(axis='y',color='#E6EBEF');ax2.set_axisbelow(True);ax2.legend(loc='lower right',frameon=False,fontsize=10)
fig.suptitle('High scores survive an interior crop, but pose robustness is incomplete',x=.07,ha='left',fontsize=20,fontweight='bold',y=.96)
foot(fig,'Left: inner radius = 80% of reviewed radius; 3 paired seeds; normals/centering retain the full-mesh convention. Concerto interior pending.\nRight: inference perturbation only; fixed seed-42 RGB models, same 64 validation scans. No rotation retraining in this comparison.')
save(fig,'03_interior_and_rotation')

# 4. Mesh reconstruction is associated with stage; nuisance controls make this concrete.
counts=pd.read_csv(SOURCE/'qc/paired_face_counts.csv');counts.to_csv(tables/'paired_face_counts.csv',index=False)
controls=pd.read_csv(SOURCE/'controls/reference_followup/comparison.csv');controls.to_csv(tables/'nuisance_controls.csv',index=False)
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,6.4));fig.subplots_adjust(left=.07,right=.96,bottom=.23,top=.78,wspace=.65)
scanners=['LABscanner','TRIOS3','TRIOS5','iTERO'];rng=np.random.default_rng(42)
for i,scanner in enumerate(scanners):
    vals=counts[counts.scanner==scanner].baseline_clean_ratio.to_numpy();median=np.median(vals)
    ax1.scatter(i+rng.uniform(-.16,.16,len(vals)),vals,s=22,alpha=.6,color=BLUE)
    ax1.plot([i-.25,i+.25],[median,median],color=ORANGE,lw=3)
    ax1.text(i,vals.max()+.07,f'median {median:.2f}×',ha='center',fontsize=10)
ax1.axhline(1,color=GRAY,linestyle='--');ax1.set_xticks(range(4),scanners);ax1.set_ylabel('Triangle count: baseline / clean')
ax1.set_ylim(.9,counts.baseline_clean_ratio.max()+.25);ax1.set_title('All 168 paired ratios exceed 1',pad=22)
ax1.set_axisbelow(True);ax1.grid(axis='y',color='#E6EBEF')
ordered=['scanner_only','face_count_only','scanner_face_count','roughness_geometry','colour_only']
controlnames=['Scanner identity','Triangle count','Scanner + triangle count','Roughness features','Global colour features']
vals=np.array([controls[controls.control==x].specimen_balanced_accuracy.iloc[0]*100 for x in ordered])
ax2.barh(range(5),vals,color=[GRAY,GRAY,ORANGE,GRAY,BLUE],height=.56)
for i,val in enumerate(vals):ax2.text(val+1.7,i,f'{val:.1f}%',va='center',fontsize=11)
ax2.set_yticks(range(5),controlnames);ax2.invert_yaxis();ax2.set_xlim(0,105);ax2.set_xticks([0,25,50,75,100])
ax2.set_xlabel('Validation specimen-balanced accuracy (%)');ax2.set_title('Cheap nuisance / feature controls',pad=22)
ax2.axvline(100/3,color=GRAY,linestyle=':');ax2.grid(axis='x',color='#E6EBEF');ax2.set_axisbelow(True)
fig.suptitle('Stage-associated mesh density needs to be separated from biological topography',x=.07,ha='left',fontsize=19,fontweight='bold',y=.96)
foot(fig,'Left: 42 development specimens × 4 scanners; scanner observations are correlated. Orange line = scanner median.\nRight: controls fit on training specimens only; same 64-scan validation cohort. Density association does not identify its biological/acquisition cause.')
save(fig,'04_mesh_density_and_controls')

# Exact original diagnostic examples, copied without image edits.
for src,name in [
    ('runs/reference_followup__pointnext_s__rgb_combined__seed42/focus/d902f623a3273e4b.png','backup_focus_PointNeXt_TRIOS3_BCG3T1-5.png'),
    ('runs/reference_followup__concerto__rgb_combined__seed42/focus/09a1ade4aca7511f.png','backup_error_Concerto_iTERO_BCG4T1-5.png')]:
    shutil.copyfile(SOURCE/src,OUT/name)

original=allruns[allruns.variant=='area_geometry'].groupby('model').agg(mean=('specimen_balanced_accuracy','mean'),seed_sd=('specimen_balanced_accuracy','std'),partial_recall=('partial_recall','mean'))
original.to_csv(tables/'original_geometry_baselines.csv')
fig,ax=plt.subplots(figsize=(10.5,5.5));fig.subplots_adjust(left=.19,right=.93,bottom=.22,top=.79)
models=['pointnext_s','tsgcnet','concerto','sonata']
for i,model in enumerate(models):
    row=original.loc[model];vals=allruns[(allruns.model==model)&(allruns.variant=='area_geometry')].specimen_balanced_accuracy.to_numpy()*100
    ax.barh(i,row['mean']*100,height=.5,color=BLUE,xerr=row.seed_sd*100,error_kw={'ecolor':DARK,'capsize':4})
    ax.scatter(vals,np.full(len(vals),i),s=20,color=DARK,zorder=3)
    ax.text(84,i,f"{row['mean']*100:.1f}%",va='center',fontweight='bold')
ax.set_yticks(range(4),['PointNeXt-S','TSGCNet','Concerto','Sonata']);ax.invert_yaxis();style(ax);ax.set_xlim(0,95)
ax.axvline(100/3,color=GRAY,ls=':')
fig.suptitle('Original geometry baselines remain modest on this validation split',x=.07,y=.96,ha='left',fontsize=18,fontweight='bold')
foot(fig,'Original area-sampled geometry protocol; mean ± seed SD, dots = 3 seeds. Backbone/head dimensions differ across models.\nSame 64 validation scans / 6 specimens. This is a model benchmark, not a matched test that geometry has no biological signal.')
save(fig,'backup_geometry_four_models')
print('Created four PNG/SVG figure pairs, two exact original diagnostics and source tables in',OUT)
