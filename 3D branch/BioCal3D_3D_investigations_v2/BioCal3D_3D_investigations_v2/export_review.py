"""Create a small results-only ZIP from the follow-up outputs (CMD compatible)."""
from pathlib import Path
from datetime import datetime
import argparse,zipfile

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'runs'/'approved_rois_v2');p.add_argument('--no-images',action='store_true');p.add_argument('--mature-only',action='store_true');a=p.parse_args()
    if a.mature_only and a.output==Path(__file__).resolve().parent/'runs'/'approved_rois_v2':a.output=a.output.with_name('approved_rois_v2_mature')
    root=a.output.resolve()
    if not (root/'followup_identity.json').exists():raise FileNotFoundError('No follow-up output found at '+str(root))
    allowed={'.csv','.json','.html','.txt','.log'}| (set() if a.no_images else {'.png','.jpg','.jpeg','.svg'})
    dest=root.parent/('3d_approved_review_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.zip');count=0
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
        for file in sorted(root.rglob('*')):
            if not file.is_file() or file.suffix.lower() not in allowed:continue
            if {'surface_cache','feature_cache'}.intersection(file.relative_to(root).parts):continue
            z.write(file,file.relative_to(root));count+=1
    print(f'Upload: {dest}\n{count} files; {dest.stat().st_size/1024**2:.1f} MiB. Original outputs unchanged.')

if __name__=='__main__':main()
