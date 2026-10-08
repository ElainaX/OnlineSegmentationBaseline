"""Post-hoc geometry/deletion evaluation. No GT is used by online execution."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

ROOT=Path('/root/autodl-tmp/output/ig_surface_deletion_20261008')
DATA=Path('/root/autodl-tmp/3RScan/3RScan_Dynamic_OnlineAnySeg')
PROBE=Path('/root/autodl-tmp/output/ig_rescan_dense_20261008/dense')
load=lambda p:json.loads(p.read_text())
results=load(ROOT/'results.json');scenes=[]
def cloud(p):return np.asarray(o3d.io.read_point_cloud(str(p)).points)
def nearest(p,q):return cKDTree(q).query(p)[0] if len(q) else np.full(len(p),np.inf)
for seq in sorted({r['seq'] for r in results}):
    records={r['variant']:r for r in results if r['seq']==seq}
    if not all(records.get(k,{}).get('status')=='complete' for k in ['off','on']):
        scenes.append(dict(seq=seq,status='incomplete',variants={k:dict(status=r['status'],error=r.get('error')) for k,r in records.items()}));continue
    off,on=ROOT/'off'/seq,ROOT/'on'/seq
    a=np.load(off/'surface_detection/bank.npz');b=np.load(on/'surface_detection/bank.npz');prior=np.load(PROBE/seq/'bank.npz')
    parity={k:np.array_equal(a[k],b[k]) and np.array_equal(a[k],prior[k]) for k in a.files}
    assert all(parity.values()),(seq,parity)
    boundary=load(DATA/seq/'session_boundary.json');cut=boundary['rescan_start_frame'];changes=boundary['changed_instances']
    removed=set(map(int,changes['removed']));changed_old=removed|set(map(int,changes['nonrigid']))|{int(r['instance_reference']) for r in changes['rigid']}
    changed_new=removed|set(map(int,changes['nonrigid']))|{int(r['instance_rescan']) for r in changes['rigid']}
    clouds=[];reference_clouds=[]
    for p in sorted((on/'surface_deletion/executor/archive').glob('*.npz')):
        pts=np.load(p)['xyz'];clouds.append(pts)
        if int(p.stem)<cut:reference_clouds.append(pts)
    deleted=np.unique(np.vstack(clouds),axis=0) if clouds else np.empty((0,3))
    gtref=cloud(DATA/'gt_reference'/seq/(seq+'_vh_clean_2.ply'));rl=np.loadtxt(DATA/'gt_seg_reference_all_things'/(seq+'.txt'),dtype=np.int64)
    assert len(gtref)==len(rl)
    distances,idx=cKDTree(gtref).query(deleted);ids=np.where((distances<=.05)&(rl[idx]>0),rl[idx]%1000,-1)
    counts=Counter(ids.tolist())
    gtnew=cloud(DATA/seq/(seq+'_vh_clean_2.ply'));nl=np.loadtxt(DATA/'gt_seg_all_things'/(seq+'.txt'),dtype=np.int64);assert len(gtnew)==len(nl)
    static=(nl>0)&~np.isin(nl%1000,list(changed_new))
    pc={mode:cloud(ROOT/mode/seq/'final.ply') for mode in ['off','on']}
    ds={mode:nearest(gtnew,p) for mode,p in pc.items()}
    coverage={}
    for tol in [.025,.05,.10]:
        x,y=ds['off']<=tol,ds['on']<=tol
        coverage[str(tol)]=dict(all_gt_off=float(x.mean()),all_gt_on=float(y.mean()),
            static_gt_vertices=int(static.sum()),static_off=float(x[static].mean()) if static.any() else None,
            static_on=float(y[static].mean()) if static.any() else None,
            static_previously_covered_lost=int((static&x&~y).sum()),static_newly_covered=int((static&~x&y).sum()))
    row=dict(seq=seq,status='complete',detector_arrays_identical_to_dense_probe=True,
             AP={mode:{k:records[mode][k] for k in ['AP','AP50','AP25','main_loop_seconds','detection_seconds']} for mode in ['off','on']},
             unique_erased_corners=len(deleted),unique_erased_before_rescan=len(np.unique(np.vstack(reference_clouds),axis=0)) if reference_clouds else 0,
             erased_near_removed_GT=sum(counts[g] for g in removed),erased_near_any_changed_GT=sum(counts[g] for g in changed_old),
             erased_near_no_change_annotation=sum(v for g,v in counts.items() if g>=0 and g not in changed_old),
             erased_unmatched=counts[-1],erased_reference_GT_counts=dict(counts),
             rescan_geometry_coverage=coverage,final_point_count={k:len(v) for k,v in pc.items()},
             deletion_summary={k:v for k,v in load(on/'surface_deletion/summary.json').items() if k!='events'})
    scenes.append(row)
report=dict(scenes=scenes,GT_used_for_audit_only=True,
            note='Erased lattice corner counts are not surface area or instance precision. Unannotated does not prove unchanged. Geometry coverage uses rescan GT. Detection is compared byte-for-byte with the fixed dense-rescan probe; deletion does not alter its metrics.')
(ROOT/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
