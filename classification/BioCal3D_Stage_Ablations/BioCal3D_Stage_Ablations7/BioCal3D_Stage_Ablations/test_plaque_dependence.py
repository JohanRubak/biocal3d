"""Connect visible plaque-like regions to actual encoder + fitted-head responses.

select: mark paired circular regions on the original image, without heatmaps.
export-head: export the fitted scaler and classifier parameters for cross-environment scoring.
run: match edit sizes, re-encode plaque/clean/random interventions, export results.
summarize: aggregate paired effects with physical specimens as the unit.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

import explain_stage_classifier as e
from images_2d import coverage_mask
from make_feature_pseudomasks import render
from study_data import seed_for, write_csv, write_json

VERSION="plaque-dependence-1.0"


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def lookup_row(run, scan):
    rows=e.read_json(run / "cohort.json")["rows"]
    found=[r for r in rows if r["scan_id"]==scan]
    if len(found)!=1:raise ValueError("Scan ID is absent or not unique; use inspect_feature_maps.py --list-scans")
    return found[0]


def distances(xy, center, shape, field):
    return np.linalg.norm((xy-np.asarray(center))*np.array([field/shape[1],field/shape[0]]),axis=1)


def pixel_xy(shape):
    yy,xx=np.mgrid[:shape[0],:shape[1]]
    return np.column_stack((xx.ravel(),yy.ravel()))


def circle_mask(shape, center, radius, field, valid):
    return (distances(pixel_xy(shape),center,shape,field)<=radius).reshape(shape)&valid


def annotation(run, row, pairs):
    rgb,valid,center,field=render(run,row)
    complete=[]
    for index,pair in enumerate(pairs,1):
        if pair.get("plaque_xy") is None and pair.get("clean_xy") is None:continue
        if pair.get("plaque_xy") is None or pair.get("clean_xy") is None:
            raise ValueError(f"Pair {index} needs both plaque-like and clean-like references")
        if not 0<pair["radius_mm"]<=3:raise ValueError("Radius must be >0 and <=3 mm")
        for role in ("plaque","clean"):
            x,y=pair[role+"_xy"]
            if not np.isfinite([x,y]).all():raise ValueError("Coordinates must be finite")
            col,r=int(round(x)),int(round(y))
            if not (0<=r<valid.shape[0] and 0<=col<valid.shape[1] and valid[r,col]):
                raise ValueError("Reference centers must be inside the original specimen")
        p=circle_mask(valid.shape,pair["plaque_xy"],pair["radius_mm"],field,valid)
        c=circle_mask(valid.shape,pair["clean_xy"],pair["radius_mm"],field,valid)
        if min(p.sum(),c.sum())<8:raise ValueError("Region contains too few rendered pixels; increase radius")
        if np.any(p&c):raise ValueError("Plaque-like and clean-like circles overlap; move them apart")
        complete.append(dict(pair_id=index,radius_mm=float(pair["radius_mm"]),
                             plaque_xy=list(map(float,pair["plaque_xy"])),clean_xy=list(map(float,pair["clean_xy"]))))
    if not complete:raise ValueError("Select at least one complete pair")
    return dict(version=VERSION,scan_id=row["scan_id"],physical_id=row["physical_id"],scanner=row["scanner"],
        recorded_stage=row["stage"],partition=row["partition"],geometry_sha256=row["geometry_sha256"],
        cohort_sha256=e.digest(run/"cohort.json"),rgb_array_sha256=array_hash(rgb),
        valid_array_sha256=array_hash(valid),image_shape=list(valid.shape),field_mm=field,pairs=complete,
        label_meaning="Human-selected visually plaque-like/clean-like regions; exploratory labels, not ground truth.")


class RegionPicker:
    def __init__(self,args,row):
        from matplotlib.widgets import Button,Slider
        self.args,self.row=args,row
        self.rgb,self.valid,_,self.field=render(args.run,row)
        self.pairs=[dict(radius_mm=args.radius_mm,plaque_xy=None,clean_xy=None)]
        self.history=[];self.role="plaque"
        self.fig,self.axes=plt.subplots(1,2,figsize=(12,6))
        self.fig.subplots_adjust(top=.79,bottom=.10,left=.03,right=.98)
        self.fig.suptitle(row["scan_id"]+" | select conservative visible regions before viewing attribution",fontsize=12)
        self.status=self.fig.text(.03,.035,"",fontsize=10)
        self.buttons=[]
        actions=[("Plaque-like",lambda _:self.set_role("plaque")),("Clean-like",lambda _:self.set_role("clean")),
                 ("Next pair",self.next_pair),("Undo",self.undo),("Save",self.save)]
        for i,(label,callback) in enumerate(actions):
            b=Button(self.fig.add_axes([.04+i*.145,.87,.13,.055]),label);b.on_clicked(callback);self.buttons.append(b)
        self.slider=Slider(self.fig.add_axes([.19,.81,.44,.025]),"Radius (mm)",.1,2.,valinit=args.radius_mm,valfmt="%.2f")
        self.fig.canvas.mpl_connect("button_press_event",self.on_click)
        self.fig.canvas.mpl_connect("key_press_event",self.on_key)
        self.refresh()

    def set_role(self,role):self.role=role;self.refresh()

    def refresh(self,message=None):
        for ax in self.axes:ax.clear();ax.imshow(self.rgb);ax.set_axis_off()
        self.axes[0].set_title("Click original image: plausible plaque / clean regions",fontsize=10)
        self.axes[1].set_title("Selected circles (not feature heatmaps)",fontsize=10)
        for i,pair in enumerate(self.pairs,1):
            for role,color in (("plaque","gold"),("clean","cyan")):
                xy=pair.get(role+"_xy")
                if xy is None:continue
                width=pair["radius_mm"]*2*self.valid.shape[1]/self.field
                height=pair["radius_mm"]*2*self.valid.shape[0]/self.field
                for ax in self.axes:
                    ax.add_patch(Ellipse(xy,width,height,fill=False,edgecolor=color,lw=2))
                    ax.text(xy[0],xy[1],str(i),color=color,fontsize=10)
        self.status.set_text(message or f"Pair {len(self.pairs)} | next click: {self.role}-like | radius locks with first click in each pair | P/C: choose role; N: next; S: save")
        self.fig.canvas.draw_idle()

    def on_click(self,event):
        if event.inaxes not in self.axes or event.xdata is None or event.button!=1:return
        x,y=float(event.xdata),float(event.ydata)
        col,r=int(round(x)),int(round(y))
        if not (0<=r<self.valid.shape[0] and 0<=col<self.valid.shape[1] and self.valid[r,col]):
            self.refresh("Click inside the specimen");return
        self.history.append(json.loads(json.dumps(self.pairs)))
        pair=self.pairs[-1]
        if pair["plaque_xy"] is None and pair["clean_xy"] is None:pair["radius_mm"]=float(self.slider.val)
        pair[self.role+"_xy"]=[x,y]
        if self.role=="plaque":self.role="clean"
        self.refresh()

    def next_pair(self,_=None):
        pair=self.pairs[-1]
        if pair["plaque_xy"] is None or pair["clean_xy"] is None:self.refresh("Finish both references before adding another pair");return
        self.history.append(json.loads(json.dumps(self.pairs)))
        self.pairs.append(dict(radius_mm=float(self.slider.val),plaque_xy=None,clean_xy=None));self.role="plaque";self.refresh()

    def undo(self,_=None):
        if self.history:self.pairs=self.history.pop()
        self.refresh()

    def save(self,_=None):
        try:
            result=annotation(self.args.run,self.row,self.pairs)
            if self.args.out.exists():raise ValueError("Selection file exists; choose a new --out")
            write_json(self.args.out,result)
            self.refresh("Saved "+str(self.args.out));print("Saved selection:",self.args.out,flush=True)
        except (OSError,ValueError) as exc:self.refresh(str(exc))

    def on_key(self,event):
        if event.key=="p":self.set_role("plaque")
        elif event.key=="c":self.set_role("clean")
        elif event.key=="n":self.next_pair()
        elif event.key=="s":self.save()
        elif event.key in ("u","backspace"):self.undo()


def matched_masks(xy,valid_native,pair,shape,field,min_count=8):
    masks={};ds={}
    for role in ("plaque","clean"):
        d=distances(xy,pair[role+"_xy"],shape,field)
        ids=np.flatnonzero((d<=pair["radius_mm"])&valid_native)
        ds[role]=(d,ids)
    count=min(len(v[1]) for v in ds.values())
    if count<min_count:raise ValueError(f"Only {count} native input units in the smaller region; increase the circle radius")
    for role,(d,ids) in ds.items():
        keep=ids[np.argsort(d[ids],kind="stable")[:count]]
        mask=np.zeros(len(xy),bool);mask[keep]=True;masks[role]=mask
    if np.any(masks["plaque"]&masks["clean"]):raise ValueError("Native reference regions overlap")
    return masks,count


def rim_distances(valid,field,xy):
    shape=valid.shape
    rim=distance_transform_edt(np.pad(valid,1),sampling=(field/shape[0],field/shape[1]))[1:-1,1:-1]
    cols=np.clip(np.rint(xy[:,0]).astype(int),0,shape[1]-1)
    rows=np.clip(np.rint(xy[:,1]).astype(int),0,shape[0]-1)
    return rim[rows,cols],rim


def random_masks(xy,valid_native,valid,pair,masks,count,field,n,seed,rim_tolerance):
    rim_values,rim=rim_distances(valid,field,xy)
    target=float(rim_values[masks["plaque"]].mean())
    candidates=np.argwhere(valid)
    rng=np.random.default_rng(seed);rng.shuffle(candidates)
    controls=[];seen=set()
    excluded=masks["plaque"]|masks["clean"]
    # Preserve the source circle's subpixel registration. Integer-only centers can
    # otherwise contain fewer pixels than a half-pixel source circle at the same radius.
    offset=np.mod(np.asarray(pair["plaque_xy"],dtype=float),1.)
    for row,col in candidates[:5000]:
        # Cheap center-distance filter before testing the actual equal-count region.
        if abs(float(rim[row,col])-target)>rim_tolerance+pair["radius_mm"]:continue
        candidate=np.array([col,row],dtype=float)+offset
        if candidate[0]>=valid.shape[1]-.5 or candidate[1]>=valid.shape[0]-.5:continue
        d=distances(xy,candidate,valid.shape,field)
        ids=np.flatnonzero((d<=pair["radius_mm"])&valid_native)
        if len(ids)<count:continue
        ids=ids[np.argsort(d[ids],kind="stable")[:count]]
        if excluded[ids].any() or abs(float(rim_values[ids].mean())-target)>rim_tolerance:continue
        key=tuple(sorted(map(int,ids)))
        if key in seen:continue
        seen.add(key)
        m=np.zeros(len(xy),bool);m[ids]=True;controls.append(m)
        if len(controls)==n:break
    if len(controls)<n:raise ValueError(f"Only {len(controls)}/{n} distinct size/rim-matched random regions found. Use smaller circles, fewer controls or a documented larger rim tolerance.")
    return controls,rim_values


def mean_max_allocation(data,result,mask,input_shape):
    if "shape" in data["meta"]:
        overlap=coverage_mask(mask.reshape(input_shape),*data["meta"]["shape"])
        coverage=data["meta"]["coverage"]
        fraction=np.divide(overlap,coverage,out=np.zeros_like(overlap),where=coverage>0).ravel().clip(0,1)
    else:fraction=mask.astype(float)
    return float(result["mean_map"]@fraction),float(result["max_map"]@fraction)


def snapshot_image(snapshot,data,valid,center,field):
    if "image" in snapshot:return snapshot["image"]
    xyz=(snapshot["coord"]*data["meta"]["scale_mm"])@data["meta"]["rotation"].T+data["meta"]["center_mm"]
    rows,cols=np.where(valid)
    target=np.column_stack(((cols+.5)*field/valid.shape[1]+center[0]-field/2,
                            center[1]+field/2-(rows+.5)*field/valid.shape[0]))
    d,ids=cKDTree(xyz[:,:2]).query(target)
    image=np.full((*valid.shape,3),127,np.uint8)
    keep=d<=.35
    image[rows[keep],cols[keep]]=np.uint8(np.clip(snapshot["color"][ids[keep]],0,1)*255)
    return image


def save_snapshot(folder,name,snapshot,data):
    if "image" in snapshot:
        Image.fromarray(snapshot["image"]).save(folder/(name+".png"))
        return
    np.savez_compressed(folder/(name+".npz"),**snapshot)
    xyz=(snapshot["coord"]*data["meta"]["scale_mm"])@data["meta"]["rotation"].T+data["meta"]["center_mm"]
    normals=snapshot["normal"]@data["meta"]["rotation"].T
    colors=np.rint(np.clip(snapshot["color"],0,1)*255)
    header=f"ply\nformat ascii 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty float nx\nproperty float ny\nproperty float nz\nend_header\n"
    with (folder/(name+".ply")).open("w",encoding="ascii",newline="\n") as f:
        f.write(header);np.savetxt(f,np.column_stack((xyz,colors,normals)),fmt="%.7f %.7f %.7f %d %d %d %.7f %.7f %.7f")


def paired_plot(folder,pair_id,fill,mode,original,mean_image,valid,xy,masks,results,snapshots,data,center,field,baseline_margin,target,against):
    fig=plt.figure(figsize=(14,7),layout="constrained")
    grid=fig.add_gridspec(2,4,height_ratios=[1,.85])
    axes=[fig.add_subplot(grid[0,i]) for i in range(4)]
    axes[0].imshow(original);axes[0].set_title("Original: selected input units",fontsize=10)
    for role,color in (("plaque","gold"),("clean","cyan")):
        points=xy[masks[role]]
        axes[0].scatter(points[:,0],points[:,1],s=2,c=color,alpha=.65)
    im=e.plot_signed(axes[1],original,valid,mean_image,"Mean allocation\nIndependent display scale")
    fig.colorbar(im,ax=axes[1],shrink=.65,label="Mean allocated contribution")
    for role,color in (("plaque","gold"),("clean","cyan")):
        points=xy[masks[role]]
        axes[1].scatter(points[:,0],points[:,1],s=4,facecolors="none",edgecolors=color,linewidths=.4,alpha=.9)
    for ax,role in zip(axes[2:],("plaque","clean")):
        ax.imshow(snapshot_image(snapshots[role],data,valid,center,field))
        ax.set_title(f"{role}-like region edited\n{int(masks[role].sum())} native input units",fontsize=10)
    for ax in axes:ax.set_axis_off()
    bar=fig.add_subplot(grid[1,:])
    labels=[r["role"] for r in results]
    values=[r["margin_drop"] for r in results]
    colors=["#db9f0e" if r["role"]=="plaque" else "#239cc4" if r["role"]=="clean" else "#94a2b1" for r in results]
    bar.bar(np.arange(len(values)),values,color=colors)
    bar.scatter(np.arange(len(values)),[r["mean_branch_margin_drop"] for r in results],marker="o",s=22,c="black",label="Mean branch")
    bar.scatter(np.arange(len(values)),[r["max_branch_margin_drop"] for r in results],marker="x",s=25,c="black",label="Max branch")
    bar.set_xticks(np.arange(len(values)),labels,rotation=15,fontsize=9);bar.axhline(0,c="black",lw=.7)
    bar.set_ylabel("Baseline margin minus margin after edit")
    bar.set_title("Positive: editing weakens target relative to comparison; negative: editing strengthens it",fontsize=10)
    bar.legend(fontsize=8)
    fig.suptitle(f"{mode} | pair {pair_id} | {fill} | {target} versus {against}\nUnedited margin {baseline_margin:+.3f}; same edit count per region; encoder rerun for every edit",fontsize=11)
    fig.savefig(folder/f"pair_{pair_id:02d}_{fill}.png",dpi=160,bbox_inches="tight");plt.close(fig)


def run_experiment(args,selection_path,out):
    selection=e.read_json(selection_path)
    row=lookup_row(args.run,selection["scan_id"])
    validated=annotation(args.run,row,selection["pairs"])
    for key in ("cohort_sha256","rgb_array_sha256","valid_array_sha256","geometry_sha256","image_shape","field_mm"):
        if selection[key]!=validated[key]:raise ValueError(f"Selection no longer matches the original scan: {key}")
    if selection["physical_id"]!=row["physical_id"]:raise ValueError("Selection specimen mismatch")
    mode=args.mode or ("dinov3_rgb" if args.encoder=="dinov3" else "concerto_rgb")
    if mode.split('_')[0]!=args.encoder:raise ValueError("Mode must match the selected encoder")
    if args.encoder=="dinov3" and args.intervention=="remove-points":raise ValueError("Point removal requires Concerto")
    target=args.target or row["stage"]
    against=args.against or ("partial" if target=="clean" else "clean")
    if target==against:raise ValueError("Target and comparison must differ")
    ti,ai=e.STAGES.index(target),e.STAGES.index(against)
    data=e.load_cache(args.run,row,mode)
    head,head_path=e.load_head(args.run,mode,args.head_source)
    attribution=e.decomposition(head,data,ti,ai)
    original,valid,center,field=render(args.run,row)
    if args.encoder=="dinov3":
        xy=pixel_xy(valid.shape);native_valid=valid.ravel();shape=valid.shape
    else:
        xy=e.native_xy(data,valid.shape,center,field)
        cols=np.clip(np.rint(xy[:,0]).astype(int),0,valid.shape[1]-1)
        rows=np.clip(np.rint(xy[:,1]).astype(int),0,valid.shape[0]-1)
        native_valid=(xy[:,0]>=-.5)&(xy[:,0]<valid.shape[1]-.5)&(xy[:,1]>=-.5)&(xy[:,1]<valid.shape[0]-.5)&valid[rows,cols]
        shape=(len(xy),)
    planned=[]
    for pair in validated["pairs"]:
        masks,count=matched_masks(xy,native_valid,pair,valid.shape,field,args.min_units)
        controls,rim=random_masks(xy,native_valid,valid,pair,masks,count,field,args.random_controls,
                                  seed_for(row["scan_id"]+str(pair["pair_id"]),args.seed),args.rim_tolerance)
        difference=abs(float(rim[masks["plaque"]].mean()-rim[masks["clean"]].mean()))
        if difference>args.rim_tolerance:
            raise ValueError(f"Pair {pair['pair_id']}: plaque/clean mean rim distances differ by {difference:.3f} mm (limit {args.rim_tolerance:.3f}). Select a cleaner control at a comparable edge distance, or explicitly document a larger --rim-tolerance.")
        masks.update({f"random_{i+1}":m for i,m in enumerate(controls)})
        planned.append((pair,masks,count,rim))
    encode=e.make_reencoder(args,row,data,mode)
    baseline=encode();baseline_margin=e.margin(head,baseline,ti,ai)
    cached_margin=e.margin(head,data["pooled"],ti,ai)
    prob=head.predict_proba(baseline[None])[0];cached_p=head.predict_proba(data["pooled"][None])[0]
    if (not np.allclose(baseline,data["pooled"],atol=1e-3,rtol=1e-3) or abs(baseline_margin-cached_margin)>.02
            or not np.allclose(prob,cached_p,atol=2e-3)):
        raise ValueError("Unedited re-encoding does not reproduce the saved stage head prediction; inspect original environments/settings")
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/"selection.json",selection)
    write_json(out/"experiment.json",dict(version=VERSION,mode=mode,scan_id=row["scan_id"],physical_id=row["physical_id"],
        scanner=row["scanner"],partition=row["partition"],target=target,against=against,intervention=args.intervention,
        fills=args.fills,random_controls=args.random_controls,rim_tolerance_mm=args.rim_tolerance,
        seed=args.seed,selection_sha256=e.digest(selection_path),head_sha256=e.digest(head_path),
        cache_sha256=data["cache_sha256"],checkpoint_sha256=e.digest(args.checkpoint),
        baseline_margin=baseline_margin,cached_margin=cached_margin,baseline_probabilities=prob.tolist(),
        unit="rendered ROI pixels" if args.encoder=="dinov3" else "area-sampled surface points",
        meaning="Input intervention sensitivity in visually selected regions; not plaque segmentation validation or proof of a biological mechanism."))
    save_snapshot(out,"original_encoder_input",encode.inspect_input(),data)
    mean_image=e.image_values(data,attribution["mean_map"],valid,center,field)
    observations=[];paired=[]
    fills=["remove"] if args.intervention=="remove-points" else args.fills
    for pair,masks,count,rim in planned:
        pair_folder=out/f"pair_{pair['pair_id']:02d}";pair_folder.mkdir()
        np.savez_compressed(pair_folder/"native_edit_masks.npz",**masks,native_xy_pixels=xy)
        for fill in fills:
            results=[];snapshots={}
            for role,mask in masks.items():
                changed=encode(mask.reshape(shape),fill)
                after=e.margin(head,changed,ti,ai)
                after_p=head.predict_proba(changed[None])[0]
                scaler=head.named_steps["standardscaler"];clf=head.named_steps["logisticregression"]
                change_terms=(clf.coef_[ti]-clf.coef_[ai])*(scaler.transform(baseline[None])[0]-scaler.transform(changed[None])[0])
                width=len(change_terms)//2
                mean_drop=float(change_terms[:width].sum());max_drop=float(change_terms[width:].sum())
                if not np.isclose(mean_drop+max_drop,baseline_margin-after,atol=2e-4,rtol=2e-5):raise ValueError("Mean/max input effects do not reconstruct the total margin change")
                mean_alloc,max_alloc=mean_max_allocation(data,attribution,mask,valid.shape)
                result=dict(scan_id=row["scan_id"],physical_id=row["physical_id"],scanner=row["scanner"],partition=row["partition"],
                    mode=mode,target=target,against=against,pair_id=pair["pair_id"],selection_sha256=e.digest(selection_path),
                    intervention=args.intervention,fill=fill,role=role,native_edit_count=count,
                    edit_fraction=float(count/native_valid.sum()),mean_rim_distance_mm=float(rim[mask].mean()),
                    mean_allocation=mean_alloc,max_allocation=max_alloc,baseline_margin=baseline_margin,
                    after_margin=after,margin_drop=baseline_margin-after,
                    mean_branch_margin_drop=mean_drop,max_branch_margin_drop=max_drop,
                    **{f"before_p_{s}":float(prob[i]) for i,s in enumerate(e.STAGES)},
                    **{f"after_p_{s}":float(after_p[i]) for i,s in enumerate(e.STAGES)})
                results.append(result);observations.append(result)
                if role in ("plaque","clean"):
                    snapshots[role]=encode.inspect_input(mask.reshape(shape),fill)
                    save_snapshot(pair_folder,f"{role}_{fill}_encoder_input",snapshots[role],data)
                print(f"{mode} pair {pair['pair_id']} {fill} {role}: margin drop {baseline_margin-after:+.3f}",flush=True)
            by_role={r["role"]:r for r in results}
            random_effects=np.array([r["margin_drop"] for r in results if r["role"].startswith("random_")])
            p,c=by_role["plaque"],by_role["clean"]
            paired.append(dict(scan_id=row["scan_id"],physical_id=row["physical_id"],scanner=row["scanner"],partition=row["partition"],
                mode=mode,target=target,against=against,pair_id=pair["pair_id"],selection_sha256=e.digest(selection_path),
                intervention=args.intervention,fill=fill,native_edit_count=count,
                plaque_margin_drop=p["margin_drop"],clean_margin_drop=c["margin_drop"],
                random_mean_margin_drop=float(random_effects.mean()),
                plaque_minus_clean=p["margin_drop"]-c["margin_drop"],
                plaque_minus_random=p["margin_drop"]-float(random_effects.mean()),
                plaque_edit_weakens_target=p["margin_drop"]>0,
                plaque_exceeds_clean=p["margin_drop"]>c["margin_drop"],
                plaque_exceeds_random_mean=p["margin_drop"]>float(random_effects.mean()),
                mean_allocation_plaque_minus_clean=p["mean_allocation"]-c["mean_allocation"],
                max_allocation_plaque_minus_clean=p["max_allocation"]-c["max_allocation"],
                mean_branch_plaque_minus_clean=p["mean_branch_margin_drop"]-c["mean_branch_margin_drop"],
                max_branch_plaque_minus_clean=p["max_branch_margin_drop"]-c["max_branch_margin_drop"],
                random_fraction_below_plaque=float(np.mean(random_effects<p["margin_drop"])),
                plaque_clean_rim_difference_mm=abs(p["mean_rim_distance_mm"]-c["mean_rim_distance_mm"])))
            paired_plot(pair_folder,pair["pair_id"],fill,mode,original,mean_image,valid,xy,masks,results,snapshots,data,center,field,baseline_margin,target,against)
            write_csv(out/"region_effects.csv",observations);write_csv(out/"paired_effects.csv",paired)
    return out


def summarize(args):
    files=[]
    for path in args.results:
        files.extend([path] if path.is_file() else sorted(path.rglob("paired_effects.csv")))
    files=list(dict.fromkeys(p.resolve() for p in files))
    if not files:raise ValueError("No paired_effects.csv files found")
    rows=[];seen=set()
    for file in files:
        with file.open(encoding="utf-8-sig",newline="") as stream:
            for row in csv.DictReader(stream):
                key=tuple(row[k] for k in ("selection_sha256","mode","target","against","intervention","fill","pair_id"))
                if key in seen:raise ValueError("Duplicate experiment rows; select one run per selection/mode/contrast/fill/pair")
                seen.add(key);rows.append(row)
    # First average ROI pairs within scan, then scanner/stage views within a physical specimen.
    fields=["plaque_minus_clean","plaque_minus_random","mean_allocation_plaque_minus_clean","max_allocation_plaque_minus_clean",
            "mean_branch_plaque_minus_clean","max_branch_plaque_minus_clean"]
    groups={}
    for row in rows:
        key=tuple(row[k] for k in ("mode","target","against","intervention","fill","physical_id","scan_id"))
        groups.setdefault(key,[]).append(row)
    scan_means={key:{f:float(np.mean([float(r[f]) for r in rr])) for f in fields} for key,rr in groups.items()}
    specimens={}
    for key,means in scan_means.items():specimens.setdefault(key[:-1],[]).append(means)
    records=[]
    for key,scans in specimens.items():
        rec=dict(zip(("mode","target","against","intervention","fill","physical_id"),key))
        rec.update({f:float(np.mean([s[f] for s in scans])) for f in fields});rec["scan_count"]=len(scans);records.append(rec)
    if args.out.exists():raise ValueError("Summary output exists; choose a new directory")
    args.out.mkdir(parents=True)
    write_csv(args.out/"by_specimen.csv",records)
    keys=sorted({tuple(r[k] for k in ("mode","target","against","intervention","fill")) for r in records})
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout="constrained")
    rng=np.random.default_rng(123);summary=[]
    for i,key in enumerate(keys):
        group=[r for r in records if tuple(r[k] for k in ("mode","target","against","intervention","fill"))==key]
        label="\n".join([key[0],key[4],key[1]+" vs "+key[2]])
        for ax,field in zip(axes[:2],fields[:2]):
            values=np.array([r[field] for r in group]);jitter=np.linspace(-.08,.08,len(values))
            ax.scatter(i+jitter,values,s=25,alpha=.8)
            mean=float(values.mean());lo=hi=None
            if len(values)>=3:
                estimates=values[rng.integers(len(values),size=(2000,len(values)))].mean(axis=1)
                lo,hi=map(float,np.quantile(estimates,[.025,.975]));ax.plot([i,i],[lo,hi],c="black",lw=1.5)
            ax.plot(i,mean,marker="_",markersize=14,c="black")
            summary.append(dict(zip(("mode","target","against","intervention","fill"),key),metric=field,
                                specimen_count=len(values),mean=mean,bootstrap_low=lo,bootstrap_high=hi))
        axes[2].scatter([r["mean_allocation_plaque_minus_clean"] for r in group],
                        [r["mean_branch_plaque_minus_clean"] for r in group],label=label.replace("\n"," / "),alpha=.8)
    labels=["\n".join([k[0],k[4],k[1]+" vs "+k[2]]) for k in keys]
    for ax,title in zip(axes[:2],["Plaque edit effect minus clean edit effect","Plaque edit effect minus random edit effect"]):
        ax.axhline(0,c="gray",lw=1);ax.set_xticks(range(len(keys)),labels,rotation=20,fontsize=8)
        ax.set_title(title,fontsize=11);ax.set_ylabel("Difference in target-vs-comparison margin drops")
    axes[2].axhline(0,c="gray",lw=1);axes[2].axvline(0,c="gray",lw=1)
    axes[2].set_xlabel("Mean-map allocation: plaque minus clean")
    axes[2].set_ylabel("Mean-branch input effect: plaque minus clean")
    axes[2].set_title("Does allocated evidence track actual input sensitivity?",fontsize=10)
    axes[2].legend(fontsize=6)
    fig.suptitle("Each dot is one physical specimen; ROI pairs and repeated scanner views averaged\nExploratory input sensitivity; bootstrap intervals shown only with >=3 specimens",fontsize=11)
    fig.savefig(args.out/"dependence_summary.png",dpi=160,bbox_inches="tight");plt.close(fig)
    write_csv(args.out/"summary.csv",summary)
    write_json(args.out/"sources.json",dict(files=[str(f) for f in files],meaning="Descriptive specimen-level bootstrap conditional on selected regions, fits and interventions; not biological causal proof."))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    select=commands.add_parser("select")
    select.add_argument("--run",type=Path,required=True);select.add_argument("--scan",required=True)
    select.add_argument("--out",type=Path,required=True);select.add_argument("--radius-mm",type=float,default=.5)
    select.add_argument("--plaque",nargs=2,type=float,metavar=("X","Y"))
    select.add_argument("--clean",nargs=2,type=float,metavar=("X","Y"))
    export = commands.add_parser("export-head", help="Run in the original head-fitting environment")
    export.add_argument("--run", type=Path, required=True)
    export.add_argument("--mode", choices=e.MODES, required=True)
    export.add_argument("--head-source", choices=["locked_test", "development_cv"], default="locked_test")
    run=commands.add_parser("run")
    run.add_argument("--run",type=Path,required=True);run.add_argument("--selection",type=Path,required=True)
    run.add_argument("--encoder",choices=["dinov3","concerto"],required=True)
    run.add_argument("--mode",choices=e.MODES)
    run.add_argument("--repo",type=Path,required=True);run.add_argument("--checkpoint",type=Path,required=True)
    run.add_argument("--snapshot",type=Path);run.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    run.add_argument("--head-source",choices=["locked_test","development_cv"],default="locked_test")
    run.add_argument("--target",choices=e.STAGES);run.add_argument("--against",choices=e.STAGES)
    run.add_argument("--intervention",choices=["appearance","remove-points"],default="appearance")
    run.add_argument("--fills",nargs="+",choices=["median","neutral"],default=["median","neutral"])
    run.add_argument("--random-controls",type=int,default=5);run.add_argument("--rim-tolerance",type=float,default=.5)
    run.add_argument("--min-units",type=int,default=8);run.add_argument("--seed",type=int,default=123)
    run.add_argument("--out",type=Path,required=True)
    summary=commands.add_parser("summarize")
    summary.add_argument("--results",nargs="+",type=Path,required=True);summary.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    try:
        if args.command=="select":
            if args.out.exists():raise ValueError("Selection output exists; choose a new file")
            if not 0<args.radius_mm<=2:raise ValueError("Radius must be >0 and <=2 mm")
            row=lookup_row(args.run,args.scan)
            if args.plaque is not None or args.clean is not None:
                if args.plaque is None or args.clean is None:raise ValueError("Provide both --plaque and --clean coordinates")
                result=annotation(args.run,row,[dict(radius_mm=args.radius_mm,plaque_xy=args.plaque,clean_xy=args.clean)])
                write_json(args.out,result)
                print("Saved selection:",args.out,flush=True)
            else:
                try:plt.switch_backend("TkAgg")
                except (ImportError,RuntimeError) as exc:raise ValueError("Desktop display unavailable; run locally with Tkinter or supply --plaque X Y --clean X Y") from exc
                picker=RegionPicker(args,row);plt.show()
        elif args.command == "export-head":
            path = e.export_head(args.run, args.mode, args.head_source)
            print("Saved portable fitted stage head:", path, flush=True)
        elif args.command=="run":
            if args.out.exists():raise ValueError("Experiment output exists; choose a new directory")
            if args.random_controls<1 or args.rim_tolerance<0 or args.min_units<1:raise ValueError("Require controls>=1, rim tolerance>=0 and min units>=1")
            if len(set(args.fills))!=len(args.fills):raise ValueError("Duplicate fill styles")
            run_experiment(args,args.selection,args.out)
            print("Saved region experiment:",args.out,flush=True)
        else:
            summarize(args)
            print("Saved specimen summary:",args.out,flush=True)
    except (OSError,ValueError,KeyError,ImportError,RuntimeError) as exc:parser.exit(2,f"Error: {exc}\n")


if __name__=="__main__":main()
