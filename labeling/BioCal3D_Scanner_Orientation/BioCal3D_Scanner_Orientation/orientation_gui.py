"""Top-view landmark and rotation review; only preview geometry is simplified."""
from pathlib import Path
import json
import math
import queue
import threading
import traceback
import uuid
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.collections import PolyCollection
from matplotlib.transforms import Affine2D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import Image
import orient_scanners as core
from mesh_io import load_mesh, mesh_appearance
from original_models import OriginalModelIndex
from biocal3d_raster import sample_texture


def preview_model(root, row, originals):
    """The full original MUST first enter the already-extracted ROI coordinate frame."""
    g = core.arrays(root/row['sample_dir']/'geometry.npz')
    hint = str(g.get('source_ply',row.get('source','')))
    checksum = str(g.get('source_sha256',''))
    path = originals.resolve([hint],scanner=row['scanner'],sample=row.get('sample_id',Path(row['sample_dir']).name),checksum=checksum)
    m = load_mesh(path)
    v = core.points(m.vertices,np.asarray(g['original_to_canonical'],float))
    f = m.faces
    # Preview ONLY: area filtering avoids drawing underside triangles over the visible top.
    tri = v[f]; cross = np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    ids = np.flatnonzero(np.abs(cross[:,2])>1e-10)
    # Keep all faces up to a generous cap; rendering is done once and then rotated as an image.
    if len(ids)>160000: ids=ids[np.linspace(0,len(ids)-1,160000,dtype=int)]
    rgb, wedge = mesh_appearance(m)
    appearance = 'vertex RGB'
    if wedge is not None and m.texture:
        tex = np.asarray(Image.open(m.texture).convert('RGB'),dtype=float)/255
        colors = sample_texture(tex,wedge.reshape(-1,3,2)[ids].mean(axis=1),True)
        appearance = 'texture RGB'
    elif rgb is not None:
        colors = rgb[f[ids]].mean(axis=1)/255
    else:
        colors = np.tile([.72,.76,.82],(len(ids),1)); appearance='geometry shading (no matched texture)'
    n = cross[ids]/np.maximum(np.linalg.norm(cross[ids],axis=1,keepdims=True),1e-30)
    shade = .55+.45*np.abs(n[:,2]); colors=np.clip(colors*shade[:,None],0,1)
    tri = tri[ids]; order=np.argsort(tri[:,:,2].mean(axis=1))
    # Generate a square image with physical millimeter extent, centered on canonical ROI origin.
    extent=float(max(np.abs(v[:,:2]).max()*1.05,7.))
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig=Figure(figsize=(7,7),dpi=140); canvas=FigureCanvasAgg(fig)
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(-extent,extent),ylim=(-extent,extent),aspect='equal');ax.axis('off')
    ax.add_collection(PolyCollection(tri[order,:,:2],facecolors=colors[order],edgecolors='none',antialiased=False))
    canvas.draw(); im=np.asarray(canvas.buffer_rgba()).copy(); fig.clear()
    return {'image':im,'extent':extent,'path':str(path),'appearance':appearance,
            'sample_id':row.get('sample_id',Path(row['sample_dir']).name),'source_sha256':core.sha(path)}


class App:
    def __init__(self,root):
        self.root=root;root.title('BioCal3D | Match scanner orientation to TRIOS5');root.geometry('1450x940')
        self.data_root=None; self.raw=None; self.rr=[];self.evidence={};self.config=None
        self.cache={};self.source=None;self.reference=None;self.pending=[];self.busy=False
        self.q=queue.Queue();self.angle=tk.DoubleVar(value=0);self.scanner=tk.StringVar();self.sample=tk.StringVar();self.ref_sample=tk.StringVar()
        self.status=tk.StringVar(value='1. Load the full approved ROI dataset.  2. Select the original full-model data folder.')
        bar=ttk.Frame(root,padding=8);bar.pack(fill='x')
        ttk.Button(bar,text='1  Load Approved ROI dataset',command=self.choose_dataset).pack(side='left',padx=4)
        ttk.Button(bar,text='2  Original full-model folder',command=self.choose_raw).pack(side='left',padx=4)
        ttk.Button(bar,text='Load saved rotations',command=self.load_saved).pack(side='left',padx=4)
        ttk.Button(bar,text='Save rotations',command=self.save).pack(side='left',padx=4)
        ttk.Button(bar,text='Apply saved rotations to dataset copies…',command=self.apply).pack(side='right',padx=4)
        selectors=ttk.Frame(root,padding=8);selectors.pack(fill='x')
        ttk.Label(selectors,text='TRIOS5 representative:').pack(side='left')
        self.rc=ttk.Combobox(selectors,textvariable=self.ref_sample,state='readonly',width=18);self.rc.pack(side='left',padx=5);self.rc.bind('<<ComboboxSelected>>',lambda e:self.load_previews())
        ttk.Label(selectors,text='Scanner to rotate:').pack(side='left')
        self.sc=ttk.Combobox(selectors,textvariable=self.scanner,state='readonly',width=14);self.sc.pack(side='left',padx=5);self.sc.bind('<<ComboboxSelected>>',lambda e:self.select_scanner())
        ttk.Label(selectors,text='Representative:').pack(side='left')
        self.pc=ttk.Combobox(selectors,textvariable=self.sample,state='readonly',width=18);self.pc.pack(side='left',padx=5);self.pc.bind('<<ComboboxSelected>>',lambda e:self.load_previews())
        ttk.Button(selectors,text='Load / refresh preview',command=self.load_previews).pack(side='left',padx=8)
        ttk.Label(root,text='Top views in the EXISTING ROI frame. Match the same teeth/holder direction; exclude plaque from your decision.',padding=5).pack()
        self.fig=Figure(figsize=(14,5),dpi=100);self.axes=self.fig.subplots(1,3)
        self.canvas=FigureCanvasTkAgg(self.fig,master=root);self.canvas.get_tk_widget().pack(fill='both',expand=True)
        NavigationToolbar2Tk(self.canvas,root)
        self.canvas.mpl_connect('button_press_event',self.click)
        controls=ttk.Frame(root,padding=8);controls.pack(fill='x')
        ttk.Label(controls,text='Rotation (degrees, counterclockwise):').pack(side='left')
        self.slider=ttk.Scale(controls,from_=-180,to=180,variable=self.angle,command=lambda v:self.draw());self.slider.pack(side='left',fill='x',expand=True,padx=8)
        spin=ttk.Spinbox(controls,from_=-180,to=180,increment=.1,textvariable=self.angle,width=9,command=self.draw);spin.pack(side='left');spin.bind('<Return>',lambda e:self.draw())
        ttk.Button(controls,text='Reset 0°',command=lambda:self.set_angle(0)).pack(side='left',padx=5)
        ttk.Button(controls,text='Match two landmarks',command=self.landmarks).pack(side='left',padx=5)
        ttk.Button(controls,text='Accept this scanner',command=self.accept).pack(side='left',padx=5)
        self.note=tk.StringVar(value='TRIOS5 stays at 0°. Use the rotation slider, or click the same ordered pair of tooth/holder features in both views.')
        ttk.Label(root,textvariable=self.note,wraplength=1380,padding=6).pack(fill='x')
        ttk.Label(root,textvariable=self.status,wraplength=1380,padding=8,relief='sunken').pack(fill='x')
        self.draw();root.after(100,self.poll)

    def task(self,func,callback):
        if self.busy:return
        self.busy=True
        def run():
            try:self.q.put(('done',callback,func()))
            except Exception as e:self.q.put(('error',str(e),traceback.format_exc()))
        threading.Thread(target=run,daemon=True).start()

    def poll(self):
        try:
            while True:
                item=self.q.get_nowait()
                if item[0]=='log':self.status.set(item[1])
                elif item[0]=='error':
                    self.busy=False;self.status.set('Operation failed; original data were not modified.');print(item[2]);messagebox.showerror('BioCal3D',item[1])
                else:self.busy=False;item[1](item[2])
        except queue.Empty:pass
        self.root.after(100,self.poll)

    def choose_dataset(self):
        if self.busy:return
        folder=filedialog.askdirectory(title='Select the FULL Approved ROI snapshot (scan_manifest.csv inside)')
        if not folder:return
        root=Path(folder).resolve();self.status.set('Reading inventory and frame mappings…')
        def work():
            if core.existing_stamp(root):raise ValueError('Select the original, unoriented approved snapshot.')
            rr=core.inventory(root)
            if not any(r['scanner']=='TRIOS5' for r in rr):raise ValueError('TRIOS5 is missing from this dataset.')
            return rr,core.frame_evidence(root,rr)
        def done(result):
            self.data_root=root;self.rr,self.evidence=result;self.cache={};self.reference=self.source=None
            self.config={'schema':'biocal3d_scanner_orientation_v1','config_id':uuid.uuid4().hex,'created_utc':core.now(),
                         'reference':'TRIOS5','method':'manually reviewed rotation about existing canonical Z; rigid; no labels used',
                         'source_dataset':str(root),'frame_evidence':self.evidence,
                         'corrections':{'TRIOS5':{'angle_deg':0.,'matrix':np.eye(4).tolist(),'reviewed':True}}}
            present=[s for s in core.SCANNERS if any(r['scanner']==s for r in self.rr)]
            sets=[{self.sample_id(r) for r in self.rr if r['scanner']==s} for s in present]
            common=sorted(set.intersection(*sets));preferred=common[0] if common else sorted(sets[0])[0]
            self.rc['values']=sorted(sets[0]);self.ref_sample.set(preferred)
            self.sc['values']=[s for s in present if s!='TRIOS5']
            if len(present)>1:self.scanner.set(self.sc['values'][0]);self.select_scanner()
            self.status.set(f'{len(self.rr)} scans loaded. Choose the original full-model folder next. Dataset: {root}')
        self.task(work,done)

    @staticmethod
    def sample_id(r):return r.get('sample_id',Path(r['sample_dir']).name)

    def select_scanner(self):
        if self.busy:return
        values=sorted({self.sample_id(r) for r in self.rr if r['scanner']==self.scanner.get()})
        self.pc['values']=values
        self.sample.set(self.ref_sample.get() if self.ref_sample.get() in values else values[0])
        saved=(self.config or {}).get('corrections',{}).get(self.scanner.get(),{})
        self.angle.set(saved.get('angle_deg',0));self.source=None
        self.load_previews()

    def choose_raw(self):
        if self.busy:return
        folder=filedialog.askdirectory(title='Select original data root containing TRIOS5 / TRIOS3 / iTERO / LABscanner')
        if folder:self.raw=OriginalModelIndex(folder);self.cache={};self.load_previews()

    def load_previews(self):
        if self.busy or not self.rr or not self.raw or not self.scanner.get():return
        ref=next(r for r in self.rr if r['scanner']=='TRIOS5' and self.sample_id(r)==self.ref_sample.get())
        src=next(r for r in self.rr if r['scanner']==self.scanner.get() and self.sample_id(r)==self.sample.get())
        self.status.set('Loading original full models and preparing top-view previews…')
        def work():
            result=[]
            for r in (ref,src):
                key=r['scan_id']
                if key not in self.cache:self.cache[key]=preview_model(self.data_root,r,self.raw)
                result.append(self.cache[key])
            return result
        def done(result):
            self.reference,self.source=result;self.pending=[]
            self.note.set('Choose matching tooth/holder features. Change representatives to check the same rotation on additional scans.')
            self.status.set('Reference: '+self.reference['path']+' | Source: '+self.source['path']);self.draw()
        self.task(work,done)

    def set_angle(self,a):self.angle.set((float(a)+180)%360-180);self.draw()

    def draw(self):
        try:a=float(self.angle.get())
        except tk.TclError:return
        for ax in self.axes:ax.clear();ax.set_aspect('equal');ax.set_xlabel('Canonical X (mm)');ax.set_ylabel('Canonical Y (mm)')
        self.axes[0].set_title('TRIOS5 — fixed reference');self.axes[1].set_title(f'{self.scanner.get() or "Scanner"} — {a:.2f}°');self.axes[2].set_title('Overlay: TRIOS5 + rotated scanner')
        extent=max([m['extent']*1.42 for m in (self.reference,self.source) if m]+[10.])
        for ax in self.axes:
            ax.set_xlim(-extent,extent);ax.set_ylim(-extent,extent);ax.axhline(0,color='#dce2e9',lw=.6,zorder=0);ax.axvline(0,color='#dce2e9',lw=.6,zorder=0)
        for m,ids,angle,alpha in [(self.reference,[0,2],0,1),(self.source,[1,2],a,.55)]:
            if not m:continue
            ex=m['extent']
            for i in ids:
                ax=self.axes[i];trans=Affine2D().rotate_deg(angle)+ax.transData
                # White background is transparent in the overlay.
                im=m['image'].copy()
                if i==2:im[:,:,3]=np.where(np.min(im[:,:,:3],axis=2)>250,0,int(255*alpha))
                ax.imshow(im,extent=(-ex,ex,-ex,ex),origin='upper',transform=trans,zorder=2)
            self.axes[ids[0]].text(.02,.02,m['sample_id']+' | '+m['appearance'],transform=self.axes[ids[0]].transAxes,fontsize=8)
        self.fig.tight_layout();self.canvas.draw_idle()

    def landmarks(self):
        if not self.source:return
        self.pending=['ref',[]];self.note.set('Click TWO distinct matching tooth/holder features in the LEFT TRIOS5 view: first A, then B. Use the same physical order for the other scanner.')

    def click(self,event):
        if not self.pending or event.xdata is None:return
        phase,pts=self.pending;expected=self.axes[0] if phase=='ref' else self.axes[1]
        if event.inaxes!=expected:return
        pts.append([event.xdata,event.ydata]);expected.plot(event.xdata,event.ydata,'o',color='#ef476f',ms=6);expected.text(event.xdata,event.ydata,str(len(pts)),color='#a00028');self.canvas.draw_idle()
        if len(pts)<2:return
        if np.linalg.norm(np.subtract(pts[1],pts[0]))<1: self.pending=[];self.note.set('Points were too close; select landmarks at least 1 mm apart.');return
        if phase=='ref':
            self.ref_points=pts;self.pending=['src',[]];self.note.set('Now click the SAME ordered pair A then B in the MIDDLE scanner view.');return
        u=np.subtract(self.ref_points[1],self.ref_points[0]);v=np.subtract(pts[1],pts[0])
        delta=math.degrees(math.atan2(u[1],u[0])-math.atan2(v[1],v[0]));self.pending=[]
        self.set_angle(self.angle.get()+delta);self.note.set('Landmark direction matched. Inspect the overlay, refine the angle if needed, then Accept this scanner.')

    def accept(self):
        if self.busy or not self.source or not self.config:return
        a=float(self.angle.get());s=self.scanner.get()
        self.config['corrections'][s]={'angle_deg':a,'matrix':core.rotation(a).tolist(),'reviewed':True,
            'reference_model':self.reference,'source_model':self.source}
        # Preview arrays are not serialized into the configuration.
        for key in ('reference_model','source_model'):self.config['corrections'][s][key]={k:v for k,v in self.config['corrections'][s][key].items() if k!='image'}
        self.config['corrections'][s]['accepted_utc']=core.now()
        self.status.set(f'Accepted {s}: {a:.3f}°. Check another representative or proceed to the next scanner; save when all are accepted.')

    def save(self):
        if self.busy or not self.config:return
        missing={r['scanner'] for r in self.rr}-set(self.config['corrections'])
        if missing:messagebox.showerror('Rotations incomplete','Accept rotations for: '+', '.join(sorted(missing)));return
        path=filedialog.asksaveasfilename(title='Save reusable orientation corrections',initialfile='scanner_rotations_TRIOS5.json',defaultextension='.json')
        if not path:return
        core.dump(path,self.config);core.load_config(path)
        preview=Path(path).with_suffix('.preview.png');self.fig.savefig(preview,dpi=160)
        self.status.set('Saved '+str(path)+'. Apply the same file to Approved ROIs and later to calibration/agreement datasets.')

    def load_saved(self):
        if self.busy:return
        path=filedialog.askopenfilename(title='Load scanner_rotations_TRIOS5.json',filetypes=[('JSON','*.json')])
        if not path:return
        try:
            c=core.load_config(path)
            if self.rr:
                core.preflight(self.data_root,c)
                self.config=c;self.set_angle(c['corrections'].get(self.scanner.get(),{}).get('angle_deg',0))
            self.status.set('Saved corrections loaded: '+str(path))
        except Exception as e:messagebox.showerror('BioCal3D',str(e))

    def apply(self):
        if self.busy:return
        path=filedialog.askopenfilename(title='Select saved scanner_rotations_TRIOS5.json',filetypes=[('JSON','*.json')])
        if not path:return
        try:c=core.load_config(path)
        except Exception as e:messagebox.showerror('BioCal3D',str(e));return
        folder=filedialog.askdirectory(title='Select ORIGINAL Approved ROI dataset, calibration/agreement folder, or their parent')
        if not folder:return
        output=filedialog.asksaveasfilename(title='Choose a NEW output folder name (not a file)',initialdir=str(Path(folder).parent),initialfile=Path(folder).name+'_TRIOS5_oriented')
        if not output:return
        self.status.set('Checking frames and approved surfaces before creating the output copy…')
        self.task(lambda:core.apply_tree(folder,output,c,lambda s:self.q.put(('log',s))),
                  lambda out:(self.status.set('Finished: '+str(out)),messagebox.showinfo('BioCal3D','Oriented copy is ready:\n'+str(out)+'\n\nOriginal files were not changed.')))

def launch():
    root=tk.Tk();App(root);root.mainloop()

if __name__=='__main__':launch()
