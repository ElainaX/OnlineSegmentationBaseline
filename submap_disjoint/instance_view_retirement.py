"""Whole-instance absence from independent views of a fixed reference surface.

Experimental alternative to demanding repeated confirmations at each exact
bank sample. Original RGB-D conflicts still trigger candidates. Subsequent
whole-object confirmation is depth/free-space based, not per-point RGB-gated.
"""
import json
import numpy as np
from .direct_retirement import DirectIGInstanceRetirement
from .readonly_detection_core import SUPPORT,CONTRADICTION
from .surface_bank import independent_rays


def update_view_votes(state,points,keys,source,codes,frame,pose,options,cfg):
    """Unknown is not absence; positive support cancels the pending episode."""
    changed=(state.get('view_source')!=tuple(source) or
             not np.array_equal(state.get('view_keys'),keys))
    if changed:
        state['view_votes']=[];state['view_source']=tuple(source);state['view_keys']=keys.copy()
    n=len(codes);free=int((codes==CONTRADICTION).sum());support=int((codes==SUPPORT).sum())
    frac=free/max(n,1);sf=support/max(n,1);purity=free/max(free+support,1)
    checks=[('reference_too_small',n>=options.min_reference_points),
            ('too_few_current_free_points',free>=options.min_confirmed_points),
            ('insufficient_whole_instance_coverage',frac>=options.min_confirmed_fraction),
            ('current_surface_support',sf<=options.max_recent_support_fraction),
            ('mixed_evidence',purity>=options.min_contradiction_purity)]
    if sf>options.max_recent_support_fraction:state['view_votes']=[]
    qualifies=all(passed for _,passed in checks);independent=False
    required=cfg['confirm_observations']
    if qualifies:
        votes=state['view_votes']
        if not votes:independent=True
        else:
            previous=votes[-1]
            center=np.median(points,axis=0).reshape(1,3)
            independent=bool(independent_rays(center,np.asarray(previous['camera_center']).reshape(1,3),
                pose[:3,3],np.array([previous['frame']]),frame,cfg)[0])
        if independent:
            votes.append(dict(frame=int(frame),camera_center=pose[:3,3].tolist(),
                              free_fraction=frac,support_fraction=sf))
            state['view_votes']=votes[-required:]
    enough=len(state['view_votes'])>=required
    reasons=[name for name,passed in checks if not passed]
    if not enough:reasons.append('insufficient_independent_whole_views')
    stats=dict(reference_points=n,confirmed_points=free if enough else 0,
        confirmed_fraction=frac if enough else 0.,support_fraction=sf,recent_support_points=support,
        contradiction_purity=purity,current_free_fraction=frac,
        independent_whole_views=len(state['view_votes']),required_whole_views=required,
        independent_view_frames=[v['frame'] for v in state['view_votes']],whole_view_votes=state['view_votes'],
        new_independent_view=independent,reference_changed=changed,reasons=reasons)
    return qualifies and enough,stats


class InstanceViewIGRetirement(DirectIGInstanceRetirement):
    def decide_candidate(self,c,confirmed,recent,codes,mature,sample,frame,pose):
        accept,stats=update_view_votes(c,c['points'][sample],c['keys'],c['source'],codes[sample],
                                     frame,pose,self.options,self.detector.cfg)
        stats.update(anchor_fraction=float(mature[sample].mean()),
                     original_point_confirmed_fraction=float(confirmed[sample].mean()),
                     original_recent_support_fraction=float(recent[sample].mean()),
                     confirmation_rule='whole-instance multi-view depth absence; not original point RGB confirmation')
        if accept:
            np.savez_compressed(self.out/f'view_confirmation_{frame:06d}_{c["id"]:05d}.npz',
                points=c['points'][sample],current_codes=codes[sample],
                view_frames=stats['independent_view_frames'],
                camera_centers=[v['camera_center'] for v in c['view_votes']],
                free_fractions=[v['free_fraction'] for v in c['view_votes']],
                support_fractions=[v['support_fraction'] for v in c['view_votes']])
        return accept,stats

    def save(self):
        report=super().save()
        report.update(method='IG independent whole-view retirement v4',
            whole_view_confirmation_count=self.detector.cfg['confirm_observations'],
            anchor_fraction_gate=False,
            whole_view_confirmation='>=60% of full reference free, <=5% supported, >=90% free among tested, >=20 free samples; repeated at independent camera rays',
            recent_support_policy='current support over threshold cancels episode; older pre-disappearance support is not a permanent veto',
            reference_change_policy='changing source IDs or reference keys resets whole-view confirmations')
        (self.out/'summary.json').write_text(json.dumps(report,indent=2));return report
