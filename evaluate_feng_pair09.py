"""Compare on identical AP evaluator and fixed G-AP-v2 evaluation domain.

Action archives are grouped by frame for comparability with existing IG-visible
results. This is the existing region-based D protocol, not official instance AP.
"""
import json,hashlib
from pathlib import Path
import numpy as np
import recompute_standard_ap as standard
from eval_gap_v2 import cloud,raw_reference,target_domain,nearest,deletion_quality,combine,VOXEL,digest

OUT=Path('/root/autodl-tmp/output/feng_pair09_20261010')
OLD=Path('/root/autodl-tmp/output/ig_visible_v3_20261010')
SEQ='pair_09_0cac7676_7272e17c'
DATA=Path('/root/autodl-tmp/3RScan/3RScan_Dynamic_Additional_20261008')

def main():
    standard.OUT=OUT;standard.SEQS=[SEQ]
    standard.ROOTS={'Feng_detect':OUT/'detect','Feng_delete':OUT/'delete'}
    ap=standard.main()
    previous=json.loads((OLD/'results.json').read_text())['sequences'][SEQ]
    boundary=json.loads((DATA/SEQ/'session_boundary.json').read_text());cut=boundary['rescan_start_frame']
    snap=OLD/'delete'/SEQ/'correction/reference_surfaces.npz'
    initial=np.load(snap)['all_points']
    ref=cloud(DATA/'gt_reference'/(SEQ+'_vh_clean_2.ply'));cur=cloud(DATA/SEQ/(SEQ+'_vh_clean_2.ply'))
    raw=Path('/root/autodl-tmp/3RScan/additional_raw_20261008')/boundary['reference_scan_id']/'labels.instances.annotated.v2.ply'
    labels,_=raw_reference(raw,ref)
    domain=target_domain(initial,ref,labels,cur,set(map(int,boundary['changed_instances']['removed'])))
    report=dict(sequence=SEQ,formula='G-AP = AP + (1-AP)*D',
        protocol='Same fixed snapshot, GT domain, tolerances and per-frame archive grouping as IG-visible-v3 results',
        domain_snapshot=str(snap),domain_sha256=digest(snap),
        caveats=['Original Feng detector runs from first frame; IG-visible starts at rescan boundary.',
                 'Single GPU run per condition; earlier IG runs show run-to-run variation.',
                 'D measures removed-object cleanup; moving objects are not rewarded.',
                 'D groups actual deletion regions by frame, not by predicted instance.'],methods={})
    for name in ('IGSubmap','IGSubmap_delete'):
        report['methods'][name]=previous['methods'][name]
    for name in ('Feng_detect','Feng_delete'):
        a=ap['sequences'][SEQ][name];root=Path(a['root'])
        summary=json.loads((root/'correction/summary.json').read_text())
        by_frame={};files=[]
        for action in summary['actions']:
            if not action.get('archive'):continue
            path=Path(action['archive']);files.append(path)
            by_frame.setdefault(action['frame'],[]).append(np.load(path)['xyz'])
        regions=[]
        for frame,arrays in sorted(by_frame.items()):
            xyz=np.concatenate(arrays)
            hit=nearest(initial,xyz)[0]<=np.sqrt(3)*VOXEL+1e-6
            regions.append(dict(key='feng_erasure_'+str(frame),frame=frame,points=initial[hit]))
        d=deletion_quality(domain,regions,cloud(root/'final.ply'),cut)
        scores={key:combine(a[key],d['score']) for key in ('AP','AP50','AP25')}
        assert all(s['AP']<=s['G_AP']<=1 for s in scores.values())
        report['methods'][name]=dict(standard_AP=a,scores=scores,deletion=d,
           executed_count=summary['unique_deleted_world_voxels'],detector_summary=summary,
           deletion_input_sha256={str(p):digest(p) for p in files})
        print(name,scores,flush=True)
    (OUT/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False))

if __name__=='__main__':main()
