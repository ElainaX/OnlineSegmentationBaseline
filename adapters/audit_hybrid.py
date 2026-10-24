"""Read-only audit of effective confirmation decisions and unchanged gates."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent
def audit_output(out):
    out=Path(out)
    assert (out/'hybrid_policy.json').read_bytes()==(R/'hybrid_policy.json').read_bytes()
    assert (out/'hybrid_adapter_sha256.txt').read_text().strip()==hashlib.sha256((R/'hybrid_confirmation.py').read_bytes()).hexdigest()
    cfg=json.loads((R.parent/'configs/detector_H.json').read_text())
    p=out/'hybrid_observations.jsonl'
    rows=[json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
    confirmations={'dual_first':0,'temporal':0}
    for row in rows:
        s=row['stats'];n=row['reference_points']
        dual=(row['depth_points']/n>=cfg['min_contradicted_fraction'] and row['rgb_points']/n>=cfg['min_contradicted_fraction'])
        effective=1 if row['hits_before']==0 and dual else 2
        assert s['hybrid_dual_current']==dual and s['hybrid_effective_confirmations']==effective
        if row['reason']=='would_remove':
            assert row['independent'] and s['tested']>=cfg['min_reference_tested_points']
            assert s['contradiction_ratio']>=cfg['contradiction_ratio']
            assert s['cumulative_tested_fraction']>=cfg['min_cumulative_tested_fraction']
            assert s['cumulative_contradicted_fraction']>=cfg['min_contradicted_fraction']
            assert row['hits_after']==row['hits_before']+1 and row['hits_after']>=effective
            confirmations['dual_first' if effective==1 else 'temporal']+=1
        if row['reason'] in ['support','cancelled']:assert row['hits_after']==0
        if row['reason'].startswith('paused_'):assert row['hits_before']==row['hits_after']
    summary=json.loads((out/'correction/summary.json').read_text())
    assert summary['detector_settings']==cfg
    return dict(verified=True,observations=len(rows),confirmations=confirmations)
