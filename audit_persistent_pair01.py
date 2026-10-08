"""Offline GT audit only; never feeds GT into retirement decisions."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from eval_global_ap import cloud,downsample

ROOT=Path('/root/autodl-tmp/output/ig_persistent_retirement_20261009')
DATA=Path('/root/autodl-tmp/3RScan/3RScan_Dynamic_OnlineAnySeg')
SEQ='pair_01_0cac767c_19f1a892'
boundary=json.loads((DATA/SEQ/'session_boundary.json').read_text())
removed=set(boundary['changed_instances']['removed'])
ref=cloud(DATA/'gt_reference'/SEQ/(SEQ+'_vh_clean_2.ply'))
labels=np.loadtxt(DATA/'gt_seg_reference_all_things'/(SEQ+'.txt'),dtype=int)
current=cloud(DATA/SEQ/(SEQ+'_vh_clean_2.ply'))
current_labels=np.loadtxt(DATA/'gt_seg_all_things'/(SEQ+'.txt'),dtype=int)
rt=cKDTree(ref);ct=cKDTree(current)

def annotate(points):
    d,idx=rt.query(points);lab=labels[idx]%1000
    ok=(d<=.05)&(labels[idx]>0)
    ids,counts=np.unique(lab[ok],return_counts=True)
    removed_hit=ok&np.isin(lab,list(removed))
    return dict(points=len(points),near_removed_reference=int(removed_hit.sum()),
        near_current_GT=int((ct.query(points)[0]<=.05).sum()),
        no_reference_label=int((~ok).sum()),
        nearest_reference_GT=[dict(id=int(i),samples=int(n),removed=bool(i in removed))
                              for i,n in sorted(zip(ids,counts),key=lambda x:-x[1])[:10]])

result={};clouds={}
for mode in ['v1_control','persistent']:
    path=ROOT/mode/'hybrid'/SEQ;folder=path/'instance_retirement'
    meta=json.loads((path/'run_metadata.json').read_text())
    summ=json.loads((folder/'summary.json').read_text())
    rows=[json.loads(l) for l in (folder/'candidate_stats.jsonl').read_text().splitlines()]
    best={}
    for row in rows:
        for c in row['candidates']:
            i=c['candidate']
            if i not in best or c['confirmed_fraction']>best[i]['confirmed_fraction']:
                best[i]=dict(c,frame=row['frame'])
    audit=[]
    for i,entry in sorted(best.items(),key=lambda x:-x[1]['confirmed_fraction']):
        if mode=='persistent':
            files=[p for p in folder.glob(f'candidate_{i:05d}_*.npz') if int(p.stem.split('_')[-1])<=entry['frame']]
            file=max(files,key=lambda p:int(p.stem.split('_')[-1]))
        else:file=folder/f'candidate_{i:05d}.npz'
        archive=np.load(file)
        audit.append(dict(**entry,GT_audit=annotate(archive['points'])))
    erased=[]
    for file in sorted((folder/'archive').glob('*.npz')):
        a=np.load(file);erased.append(dict(archive=str(file),GT_audit=annotate(a['coords'][a['erase']])))
    cloudpoints=cloud(path/'final.ply');clouds[mode]=cloudpoints
    gt,first,_=downsample(current[current_labels>0],.025,current_labels[current_labels>0])
    gtlabels=current_labels[current_labels>0][first]
    near=cKDTree(cloudpoints).query(gt)[0]<=.05
    result[mode]=dict(main_loop_seconds=meta['main_loop_seconds'],module_seconds=summ['seconds'],
        peak_torch_bytes=meta['torch_peak_allocated_bytes'],summary=summ,best_candidate_states=audit,whole_erased_regions=erased,
        geometry_GT_coverage=float(near.mean()),
        geometry_per_GT=[dict(id=int(g),points=int((gtlabels==g).sum()),covered=int(near[gtlabels==g].sum())) for g in np.unique(gtlabels)])
base=downsample(clouds['v1_control'],.025)[0]
missing=cKDTree(clouds['persistent']).query(base)[0]>.05
result['v2_missing_relative_to_control']=annotate(base[missing])
result['limitations']='5cm nearest-surface GT association is approximate. Missing relative to control is net output difference, not necessarily a causal deletion action. Shared GPU timing is not a controlled benchmark.'
(ROOT/'audit.json').write_text(json.dumps(result,indent=2))
for mode in ['v1_control','persistent']:
    r=result[mode]
    print(mode,dict(retired=r['summary']['retired_instances'],candidates=r['summary']['candidates_created'],
        main_seconds=r['main_loop_seconds'],module_seconds=r['module_seconds'],GT_coverage=r['geometry_GT_coverage'],
        best_confirmed_fraction=r['best_candidate_states'][0]['confirmed_fraction'] if r['best_candidate_states'] else None))
