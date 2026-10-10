import json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
from eval_gap_v2 import cloud,raw_reference

OUT=Path('/root/autodl-tmp/output/feng_pair09_20261010')
SEQ='pair_09_0cac7676_7272e17c'
DATA=Path('/root/autodl-tmp/3RScan/3RScan_Dynamic_Additional_20261008')

def main():
    b=json.loads((DATA/SEQ/'session_boundary.json').read_text())
    ref=cloud(DATA/'gt_reference'/(SEQ+'_vh_clean_2.ply'))
    labels,_=raw_reference(Path('/root/autodl-tmp/3RScan/additional_raw_20261008')/b['reference_scan_id']/'labels.instances.annotated.v2.ply',ref)
    tree=cKDTree(ref);removed=set(map(int,b['changed_instances']['removed']))
    report={}
    def counts(points):
        d,j=tree.query(points)
        return {str(k):int(v) for k,v in Counter(labels[j[d<=.05]]).items() if k>0}
    for mode in ('detect','delete'):
        root=OUT/mode/SEQ;side=root/'remove_detection'
        s=json.loads((root/'correction/summary.json').read_text())
        rejects=Counter();post=Counter();reasons=Counter();candidates=[]
        for line in (side/'events.jsonl').open():
            row=json.loads(line)
            if row['event']=='candidate_rejected':
                rejects.update(row.get('failed_conditions',[row.get('reason','unknown')]))
                if row['frame']>=b['rescan_start_frame']:post.update(row.get('failed_conditions',[row.get('reason','unknown')]))
            if row['event']=='candidate_observation':reasons[row['reason']]+=1
        for c in s['candidates']:
            directory=side/'candidates'/f"{c['candidate_id']:04d}"
            p=next(directory.glob('*_created.npz'));data=np.load(p)
            cover=counts(data['reference_points'])
            candidates.append(dict(**c,GT_reference_samples=cover,
                removed_GT_stage_hits=[g for g in sorted(removed) if cover.get(str(g),0)>=3]))
        action_rows=[]
        for a in s['actions']:
            cover=counts(np.load(a['archive'])['xyz']) if a.get('archive') else {}
            action_rows.append(dict(**a,GT_archive_corners=cover,
                removed_GT_archive_hits=[g for g in sorted(removed) if cover.get(str(g),0)>=3]))
        report[mode]=dict(candidates=candidates,actions=action_rows,
            candidate_rejection_reasons=dict(rejects),post_rescan_rejections=dict(post),
            observation_reasons=dict(reasons),skips=s['skips'],
            removed_GT_ids=sorted(removed),evaluated_frames=s['evaluated_frames'],
            note='3 samples/corners within 5cm is a diagnostic spatial hit, not whole-instance correctness or deletion coverage.')
    (OUT/'detection_audit.json').write_text(json.dumps(report,indent=2))
    for mode,r in report.items():
        print(mode,'candidates',len(r['candidates']),'would_remove',sum(c['state']=='would_remove' for c in r['candidates']),
              'actions',len(r['actions']),'rejects',r['post_rescan_rejections'],flush=True)

if __name__=='__main__':main()
