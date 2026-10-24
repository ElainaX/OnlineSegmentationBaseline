"""T1-any: one valid owned contradiction immediately retires the whole instance.

The frozen detector file is never edited. Exactly four candidate admission
blocks are removed from its in-memory _step, and the new-candidate call receives
the original owned pixel statistics. Reference sampling remains diagnostic and
cannot hide a valid pixel trigger. Frame/point validity, ownership, RGB/depth
classification, instance reliability, and the original whole eraser are intact.
"""
import hashlib, inspect, json, textwrap
from pathlib import Path
import numpy as np

R=Path(__file__).resolve().parent

def observe(candidate,frame,pose,codes,sources,cfg,trigger_stats=None):
    import feng_remove_detection as d
    codes,sources=np.asarray(codes),np.asarray(sources)
    stats=d.evidence_stats(codes,sources=sources)
    trigger=dict(trigger_stats if trigger_stats is not None else stats)
    candidate.last_frame=frame
    bad=codes==d.CONTRADICTION
    candidate.tested |= (codes==d.SUPPORT)|bad
    candidate.contradicted |= bad
    candidate.depth_contradicted |= bad & ((sources&d.SOURCE_DEPTH)!=0)
    candidate.rgb_contradicted |= bad & ((sources&d.SOURCE_RGB)!=0)
    for key,arr in [('tested',candidate.tested),('contradicted',candidate.contradicted),('depth',candidate.depth_contradicted),('rgb',candidate.rgb_contradicted)]:
        stats['cumulative_'+key+'_fraction']=float(arr.mean()) if len(arr) else 0.
    if candidate.state!='suspected':reason='terminal'
    elif trigger['contradiction']>0:
        candidate.hits=1;candidate.support_hits=0
        candidate.last_observation_frame=frame;candidate.last_observation_pose=pose.copy()
        candidate.state='would_remove';candidate.confirmed_frame=frame
        reason='would_remove'
    else:reason='no_valid_contradiction'
    stats['immediate_trigger']=trigger
    return reason,stats

def transformed_step(detector):
    source=Path(detector.__file__).read_bytes()
    expected=json.loads((R/'core_audit.json').read_text())['core_sha256']['feng_remove_detection.py']
    assert hashlib.sha256(source).hexdigest()==expected,'Unexpected detector source'
    text=textwrap.dedent(inspect.getsource(detector.RemoveDetector._step))
    # Remove instance-level aggregation gates, never point/frame validity checks.
    for start,end in [
        ('        pixel_failures = []','        raw_ids ='),
        ("        if len(self.candidates) >= self.cfg['max_candidates']:",'        reference, reference_colors ='),
        ("        if len(reference) < self.cfg['min_reference_points']:",'        codes, candidate_sources ='),
        ('        ref_stats = evidence_stats','        candidate = Candidate')]:
        assert text.count(start)==1,(start,end)
        assert text[text.index(start):].count(end)==1,(start,end)
        a=text.index(start);b=text.index(end,a);text=text[:a]+text[b:]
    call='        reason, stats = candidate.observe(frame, pose, codes, candidate_sources, self.cfg)'
    assert text.count(call)==2
    a=text.rindex(call)
    text=text[:a]+text[a:].replace(call,call[:-1]+', trigger_stats=pixel_stats)',1)
    namespace=dict(vars(detector));exec(compile(text,str(R/'immediate_step.generated.py'),'exec'),namespace)
    return namespace['_step'],text

def install(out):
    import feng_remove_detection as d
    assert not getattr(d.Candidate,'_immediate_installed',False)
    step,text=transformed_step(d)
    d.RemoveDetector._step=step
    global GENERATED_STEP
    GENERATED_STEP=text
    out=Path(out)
    def recorded(candidate,frame,pose,codes,sources,cfg,trigger_stats=None):
        reason,stats=observe(candidate,frame,pose,codes,sources,cfg,trigger_stats)
        row=dict(frame=int(frame),candidate_id=int(candidate.candidate_id),reason=reason,
                 reference_points=len(candidate.points),stats=stats,trigger_origin='owned_valid_pixels' if trigger_stats is not None else 'reference_points')
        with (out/'immediate_observations.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        return reason,stats
    d.Candidate.observe=recorded;d.Candidate._immediate_installed=True

def persist_policy(out):
    out=Path(out)
    if out.exists():
        (out/'immediate_step.generated.py').write_text(GENERATED_STEP)
        (out/'immediate_policy.json').write_bytes((R/'immediate_policy.json').read_bytes())
        (out/'immediate_adapter.sha256').write_text(hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

def audit(out):
    out=Path(out)
    assert (out/'immediate_policy.json').read_bytes()==(R/'immediate_policy.json').read_bytes()
    assert (out/'immediate_adapter.sha256').read_text()==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    events=[json.loads(x) for x in (out/'immediate_observations.jsonl').read_text().splitlines()] if (out/'immediate_observations.jsonl').exists() else []
    confirmed={}
    for e in events:
        t=e['stats']['immediate_trigger'];positive=t['contradiction']>0
        assert (e['reason']=='would_remove')==positive,e
        assert e['trigger_origin']=='owned_valid_pixels'
        if positive:confirmed[e['candidate_id']]=e['frame']
    summary=json.loads((out/'correction/summary.json').read_text())
    assert summary['deletion_enabled'] and summary['ground_truth_used'] is False
    assert len(summary['candidates'])==len(confirmed)==len(summary['actions'])
    for c in summary['candidates']:
        assert c['state']=='would_remove' and c['confirmations']==1
        assert c['first_frame']==c['confirmed_frame']==confirmed[c['candidate_id']]
    for a in summary['actions']:
        assert a['frame']==confirmed[a['candidate_id']]
        assert a['status'] in ['applied','no_live_descendant']
        if a['status']=='applied':
            assert a['removed_instance_records']==len(a['resolved_slots'])
            assert a['exclusive_nodes']+a['shared_nodes_preserved']==a['target_nodes']
    return dict(verified=True,observations=len(events),immediate_confirmations=len(confirmed),single_point_triggers=sum(e['stats']['immediate_trigger']['contradiction']==1 for e in events))
