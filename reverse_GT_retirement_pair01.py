"""GT-first audit of removed objects, raw confirmations, candidate gates and cleanup.

Offline only. No new inference, no change to G-AP or any online threshold.
One detector/candidate point is assigned to at most one reference GT object.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from eval_global_ap import cloud,downsample

SEQ='pair_01_0cac767c_19f1a892'
DATA=Path('/root/autodl-tmp/3RScan/3RScan_Dynamic_OnlineAnySeg')
ROOT=Path('/root/autodl-tmp/output/ig_persistent_retirement_20261009')
RUN=ROOT/'persistent/hybrid'/SEQ
OUT=ROOT/'reverse_GT';OUT.mkdir(exist_ok=True)
VOXEL=.025;TOL=.05


def coverage(target,points):
    if not len(target) or not len(points):return np.zeros(len(target),bool)
    return cKDTree(points).query(target)[0]<=TOL


def fraction(n,d):return float(n/d) if d else None


def main():
    boundary=json.loads((DATA/SEQ/'session_boundary.json').read_text())
    report=json.loads((DATA/SEQ/'gt_report.json').read_text())
    removed=sorted(map(int,boundary['changed_instances']['removed']))
    names={int(x['reference']['object_id']):x['reference'] for x in report['changed_instances']['removed']}
    rawref=cloud(DATA/'gt_reference'/SEQ/(SEQ+'_vh_clean_2.ply'))
    rawlabels=np.loadtxt(DATA/'gt_seg_reference_all_things'/(SEQ+'.txt'),dtype=int)
    ref,first,_=downsample(rawref,VOXEL,rawlabels);labels=rawlabels[first];objects=labels%1000
    initialpath=Path('/root/autodl-tmp/output/ig_correction_20261008/delete')/SEQ/'correction/reference_surfaces.npz'
    initial=downsample(np.load(initialpath)['all_points'],VOXEL)[0]
    finalGT=cloud(DATA/SEQ/(SEQ+'_vh_clean_2.ply'))
    initial_d,witness=cKDTree(initial).query(ref)
    built=initial_d<=TOL;reused=cKDTree(finalGT).query(ref)[0]<=TOL
    eligible=(labels>0)&np.isin(objects,removed)&built&~reused
    rt=cKDTree(rawref)

    def ownership(points):
        d,near=rt.query(points)
        return np.where((d<=TOL)&(rawlabels[near]>0),rawlabels[near]%1000,-1)

    bank=np.load(RUN/'surface_detection/bank.npz')
    bp=bank['points'];bowner=ownership(bp)
    active=np.zeros(len(bp),bool);ever=np.zeros(len(bp),bool)
    ever[bank['event_indices'][bank['event_states']]]=True
    meta=json.loads((RUN/'run_metadata.json').read_text())
    required=int(meta['config']['remove_detection']['confirm_observations'])
    rowdata={g:dict(GT_id=g,label=names[g]['label'],raw_vertices=names[g]['vertex_count'],
        total_reference_samples=int(((labels>0)&(objects==g)).sum()),
        initially_built=int(((labels>0)&(objects==g)&built).sum()),
        excluded_reused=int(((labels>0)&(objects==g)&built&reused).sum()),
        target_samples=int((eligible&(objects==g)).sum())) for g in removed}
    targets={g:ref[eligible&(objects==g)] for g in removed}
    for g,row in rowdata.items():
        t=targets[g];same=bowner==g
        row.update(ever_confirmed_coverage=int(coverage(t,bp[same&ever]).sum()),
                   mature_history_coverage=int(coverage(t,bp[same&(bank['support']>=required)]).sum()),
                   final_raw_confirmed_coverage=int(coverage(t,bp[same&bank['active']]).sum()),
                   peak_simultaneous_coverage=0,peak_simultaneous_frame=None)
    # Actual activation/revocation events, not the union of non-simultaneous votes.
    cursor=0;frames=bank['event_frames'];indices=bank['event_indices'];states=bank['event_states']
    assert np.all(frames[1:]>=frames[:-1])
    for frame in meta['detector_frame_ids']:
        end=int(np.searchsorted(frames,frame,side='right'))
        for j in range(cursor,end):active[indices[j]]=states[j]
        cursor=end
        for g,row in rowdata.items():
            n=int(coverage(targets[g],bp[active&(bowner==g)]).sum())
            if n>row['peak_simultaneous_coverage']:
                row['peak_simultaneous_coverage']=n;row['peak_simultaneous_frame']=frame
    np.testing.assert_array_equal(active,bank['active'])

    folder=RUN/'instance_retirement'
    best={};last={}
    for line in (folder/'candidate_stats.jsonl').read_text().splitlines():
        r=json.loads(line)
        for c in r['candidates']:
            cid=c['candidate'];entry=dict(c,frame=r['frame']);last[cid]=entry
            if cid not in best or c['confirmed_fraction']>best[cid]['confirmed_fraction']:best[cid]=entry
    candidates=[]
    for path in sorted(folder.glob('pending_*.npz')):
        a=np.load(path);cid=int(path.stem.split('_')[-1]);points=a['points'];own=ownership(points)
        ok=a['anchors']>=0;confirmed=np.zeros(len(points),bool)
        confirmed[ok]=(a['local_streak'][ok]>=required)&bank['active'][a['anchors'][ok]]
        candidates.append(dict(id=cid,points=points,owner=own,confirmed=confirmed,
                               best=best[cid],last=last[cid],anchors=a['anchors'],added_frame=a['added_frame']))

    first_confirm=np.full(len(bp),np.iinfo(np.int64).max,np.int64)
    np.minimum.at(first_confirm,bank['event_indices'][bank['event_states']],bank['event_frames'][bank['event_states']])
    first_candidate=np.full(len(bp),np.iinfo(np.int64).max,np.int64)
    for c in candidates:
        ok=c['anchors']>=0
        np.minimum.at(first_candidate,c['anchors'][ok],c['added_frame'][ok])
    first_local_erase=np.full(len(bp),np.iinfo(np.int64).max,np.int64)
    for p in sorted((RUN/'surface_deletion').glob('evidence_*.npz')):
        a=np.load(p);f=int(p.stem.split('_')[-1])
        np.minimum.at(first_local_erase,a['surface_ids'],f)

    gap=json.loads((ROOT/'gap/results.json').read_text())['variants']['persistent']
    deletion_scores={x['GT_id']:x for x in gap['deletion']['per_GT']}
    deletion_domain={x['GT_id']:x for x in gap['deletion']['per_removed_GT_domain']}
    finalpoints=cloud(RUN/'final.ply')
    # Baseline final geometry vs current to distinguish natural clearing.
    off=cloud(Path('/root/autodl-tmp/output/ig_surface_deletion_20261008/off')/SEQ/'final.ply')
    d_final=cKDTree(finalpoints).query(initial)[0];d_off=cKDTree(off).query(initial)[0]
    for g,row in rowdata.items():
        n=row['target_samples'];t=targets[g]
        row['exclusion_reason']=None if n else ('excluded_from_all_things_labels' if row['raw_vertices'] and not row['total_reference_samples'] else 'no_initial_nonreused_target_surface')
        related=[];confirmed_points=[]
        for c in candidates:
            same=c['owner']==g
            if same.any():confirmed_points.append(c['points'][same&c['confirmed']])
            if not n:continue
            covered=int(coverage(t,c['points'][same]).sum())
            if not covered:continue
            gt_coverage=covered/n
            purity=float(same.mean())
            score=2*gt_coverage*purity/(gt_coverage+purity) if purity else 0.
            related.append(dict(candidate=c['id'],target_covered=covered,target_coverage=gt_coverage,
                reference_samples=len(c['points']),same_GT_samples=int(same.sum()),reference_GT_purity=purity,
                unmatched_reference_samples=int((c['owner']<0).sum()),diagnostic_match_F1=score,
                final_confirmed_on_same_GT=int((same&c['confirmed']).sum()),
                actual_best_gate_state=c['best'],actual_last_gate_state=c['last']))
        related.sort(key=lambda x:-x['diagnostic_match_F1'])
        row['related_candidates']=related
        row['best_candidate']=related[0] if related else None
        pts=np.vstack(confirmed_points) if confirmed_points else np.empty((0,3))
        row['final_candidate_confirmed_coverage']=int(coverage(t,pts).sum())
        # Track actual detector point IDs, not merely nearby geometry. This
        # identifies confirmation that never entered any candidate's anchor set.
        close=coverage(bp,t) if n else np.zeros(len(bp),bool)
        selected=np.flatnonzero(ever&(bowner==g)&close)
        present=first_candidate[selected]<np.iinfo(np.int64).max
        in_candidate=selected[present]
        row['evidence_transfer']=dict(
            confirmed_bank_samples=len(selected),samples_in_candidate_anchor_sets=int(present.sum()),
            target_coverage_of_transferred_samples=int(coverage(t,bp[in_candidate]).sum()),
            samples_never_registered=int((~present).sum()),
            registered_after_first_confirmation=int((first_candidate[in_candidate]>first_confirm[in_candidate]).sum()),
            local_erasure_before_registration=int((first_local_erase[in_candidate]<first_candidate[in_candidate]).sum()),
            first_confirmation_frame=int(first_confirm[selected].min()) if len(selected) else None,
            first_explicit_local_erasure_frame=int(first_local_erase[selected].min())
                if len(selected) and first_local_erase[selected].min()<np.iinfo(np.int64).max else None)
        gtmask=eligible&(objects==g)
        row['final_missing_initial_witnesses']=int((d_final[witness[gtmask]]>TOL).sum())
        row['off_missing_initial_witnesses']=int((d_off[witness[gtmask]]>TOL).sum())
        row['G_AP_deleted_samples']=deletion_domain.get(g,{}).get('absent_final_samples',0)
        row['G_AP_deletion_IoU']=deletion_scores.get(g,{}).get('IoU')
        for k in ['ever_confirmed_coverage','peak_simultaneous_coverage','final_raw_confirmed_coverage',
                  'final_candidate_confirmed_coverage','G_AP_deleted_samples','mature_history_coverage']:
            row[k+'_ratio']=fraction(row[k],n)
    evaluated=[r for r in rowdata.values() if r['target_samples']]
    denominator=sum(r['target_samples'] for r in evaluated)
    assert denominator==gap['deletion']['GT_weight'],'Audit domain must equal fixed G-AP target domain'
    for r in evaluated:
        n=r['target_samples']
        assert 0<=r['final_raw_confirmed_coverage']<=r['ever_confirmed_coverage']<=n
        assert 0<=r['peak_simultaneous_coverage']<=r['ever_confirmed_coverage']
        assert r['final_candidate_confirmed_coverage']<=r['final_raw_confirmed_coverage']
        assert 0<=r['G_AP_deleted_samples']<=n
    metrics=['ever_confirmed_coverage','peak_simultaneous_coverage','final_raw_confirmed_coverage',
             'final_candidate_confirmed_coverage','G_AP_deleted_samples']
    result=dict(GT_used_only_offline=True,seq=SEQ,voxel_size_m=VOXEL,tolerance_m=TOL,
        definition='GT-first fixed G-AP target surface domain; detector/candidate samples uniquely assigned to nearest reference GT within tolerance; no forced one-to-one candidate pairing',
        matching='Rank positive GT-candidate pairs by harmonic mean of target coverage and candidate reference GT purity; diagnostic match score, NOT AP, NOT an online decision',
        limitations=['GT assignment is nearest-surface approximation, not exact identity proof.',
            'Ever-confirmed coverage can combine different times; simultaneous peak is also reported.',
            'Sum of per-GT peaks need not have occurred at one common frame.',
            'Candidate geometry match uses the final frozen ledger; best gate state can precede later lineage extensions.',
            'Final raw/candidate confirmation differs from historical confirmation and actual deletion.',
            'All-things labels exclude at least wall GT 6; this is an evaluation-domain exclusion, not missing raw GT.',
            'Actual missing surface includes natural TSDF clearing, not only explicit deletion actions.'],
        removed_GT_count=len(removed),evaluated_removed_GT_count=len(evaluated),target_weight=denominator,
        totals={k:dict(samples=sum(r[k] for r in evaluated),ratio=sum(r[k] for r in evaluated)/denominator) for k in metrics},
        any_detection_GT_count=sum(r['ever_confirmed_coverage']>0 for r in evaluated),
        candidate_matched_GT_count=sum(bool(r['best_candidate']) for r in evaluated),
        whole_retired=json.loads((folder/'summary.json').read_text())['retired_instances'],
        G_AP=gap['G_AP'],G_AP_deletion_score=gap['deletion']['score'],rows=list(rowdata.values()))
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ['removed_GT_count','evaluated_removed_GT_count','target_weight','totals','any_detection_GT_count','candidate_matched_GT_count','whole_retired']},indent=2))
    for r in evaluated:
        c=r['best_candidate']
        print(r['GT_id'],r['label'],r['target_samples'],r['ever_confirmed_coverage'],r['peak_simultaneous_coverage'],
              c['candidate'] if c else None,round(c['actual_best_gate_state']['confirmed_fraction'],6) if c else None,
              r['G_AP_deleted_samples'],r['G_AP_deletion_IoU'])
    return result


if __name__=='__main__':main()
