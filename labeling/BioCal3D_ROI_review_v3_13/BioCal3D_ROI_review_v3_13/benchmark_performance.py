"""Synthetic IO benchmark. Does not open or modify any experimental dataset."""
import argparse,importlib.util,json,statistics,sys,tempfile,time
from pathlib import Path
import numpy as np
# Exclude one-time dependency import overhead from the measured operations.
import plyfile
APP=Path(__file__).resolve().parent
sys.path.insert(0,str(APP/'biocal3d_workflow/annotator'))
import mesh_io

def measure(io,n,repeats):
    x,y=np.meshgrid(np.linspace(-7,7,n),np.linspace(-7,7,n))
    vertices=np.c_[x.ravel(),y.ravel(),(.002*(x*x+y*y)).ravel()]
    a=np.arange(n*n).reshape(n,n)[:-1,:-1].ravel()
    faces=np.vstack([np.c_[a,a+1,a+n],np.c_[a+1,a+n+1,a+n]])
    mesh=io.Mesh(Path('synthetic'),vertices,faces,io.polydata(vertices,faces,np.tile([200,60,100],(len(vertices),1))),np.ones(len(faces),bool))
    labels=np.tile(np.array([-2,-1,0,1],np.int8),int(np.ceil(len(faces)/4)))[:len(faces)]
    timings={name:[] for name in ['label_color_ply','original_color_ply','read_original_color_ply']}
    with tempfile.TemporaryDirectory(prefix='biocal3d_perf_') as tmp:
        root=Path(tmp)
        for repeat in range(repeats):
            for name,operation in [
                ('label_color_ply',lambda:io.export_label_colors(mesh,labels,root/'colors.ply')),
                ('original_color_ply',lambda:io.export_ply(mesh,labels,root/'scanner.ply')),
                ('read_original_color_ply',lambda:io.load_mesh(root/'scanner.ply'))]:
                start=time.perf_counter();operation();timings[name].append(time.perf_counter()-start)
    return dict(faces=len(faces),vertices=len(vertices),repeats=repeats,
                seconds={k:round(statistics.median(v),6) for k,v in timings.items()},
                individual_seconds=timings)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--side',type=int,default=400)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--baseline',type=Path,help='Optional older app root or annotator directory')
    parser.add_argument('--output',type=Path,default=APP/'performance_report.json')
    args=parser.parse_args()
    if args.side<3 or args.repeats<1:parser.error('side >= 3 and repeats >= 1 are required')
    report=dict(version='3.13',kind='synthetic IO only; not Windows interaction latency',updated=measure(mesh_io,args.side,args.repeats))
    if args.baseline:
        path=args.baseline/'mesh_io.py'
        if not path.exists():path=args.baseline/'biocal3d_workflow/annotator/mesh_io.py'
        spec=importlib.util.spec_from_file_location('baseline_mesh_io',path);old=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=old;spec.loader.exec_module(old)
        report['baseline']=measure(old,args.side,args.repeats)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('baseline','updated')},indent=2))
    for name in ['baseline','updated']:
        if name in report:print(name,report[name]['faces'],'triangles',report[name]['seconds'])
    print('Report:',args.output)
if __name__=='__main__':main()
