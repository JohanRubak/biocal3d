"""Install updated code into the existing app; keep environments/data/labels."""
from pathlib import Path
from datetime import datetime
import argparse,json,shutil

FILES=[
 'biocal3d_workflow/roi_geometry.py',
 'biocal3d_workflow/biocal3d_roi_core.py',
 'biocal3d_workflow/biocal3d_prepare_all.py',
 'biocal3d_workflow/rebuild_reviewed_snapshot.py',
 'biocal3d_workflow/audit_roi_geometry.py',
 'biocal3d_workflow/analyse_labeling_agreement.py',
 'biocal3d_workflow/annotator/mesh_io.py',
 'biocal3d_workflow/annotator/biocal3d_annotator.py',
 'biocal3d_workflow/annotator/annotation_features.py',
 'biocal3d_workflow/annotator/export_annotations.py',
 'biocal3d_workflow/annotator/roi_review_store.py',
 'biocal3d_workflow/annotator/roi_review_ui.py',
 'biocal3d_workflow/annotator/source_correspondence.py',
 'biocal3d_workflow/annotator/surface_display.py',
 'biocal3d_workflow/annotator/rotation_keys.py',
 'biocal3d_workflow/annotator/test_roi_review.py',
 'biocal3d_workflow/annotator/test_roi_ui.py',
 'biocal3d_workflow/annotator/test_quick_roi_review.py',
 'biocal3d_workflow/annotator/test_review_v31.py',
 'biocal3d_workflow/annotator/test_ui.py',
 'biocal3d_workflow/annotator/test_roi_rebuild.py',
 'biocal3d_workflow/annotator/test_preprocess_surface.py',
 'audit_geometry.cmd','rebuild_reviewed.cmd','README.txt','ROI_REVIEW_GUIDE.md','VALIDATION.txt',
]

def install(target):
    source=Path(__file__).resolve().parent;target=Path(target).expanduser().resolve()
    if target==source:raise ValueError('Select your EXISTING BioCal3D_Annotator app folder, not this update folder.')
    if not (target/'biocal3d_workflow/annotator/biocal3d_annotator.py').is_file():
        raise ValueError('This is not the app root. Choose the folder containing start_labelling.bat and biocal3d_workflow.')
    for name in FILES:
        if not (source/name).is_file():raise ValueError('Update is incomplete: '+name)
    backup=target/'code_backups'/('before_roi_review_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    existing=[];added=[]
    for name in FILES:
        dest=target/name
        if dest.exists():
            saved=backup/name;saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dest,saved);existing.append(name)
        else:added.append(name)
    backup.mkdir(parents=True,exist_ok=True)
    (backup/'restore_manifest.json').write_text(json.dumps(dict(replaced=existing,added=added),indent=2))
    try:
        for name in FILES:
            dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,dest)
    except Exception:
        for name in existing:shutil.copy2(backup/name,target/name)
        for name in added:
            if (target/name).is_file():(target/name).unlink()
        raise
    print('ROI review update installed. Start your usual start_labelling.bat.')
    print('Previous code backed up to:',backup)
    return backup

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('target',type=Path);a=p.parse_args();install(a.target)
