"""Check immutable core source and optionally validate saved 110-pair formulas."""
import argparse
import ast
import json
from pathlib import Path
import numpy as np
import metrics as m


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison',type=Path,help='Optional historical all_instances comparison.json')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    provenance=m.read(m.ROOT/'docs/current_source_provenance.json')
    for name,h in provenance['core_sha256'].items():
        assert m.sha(m.ROOT/name)==h, ('Core source changed',name)
    report=dict(core_files_verified=len(provenance['core_sha256']),
                ap_evaluator_sha256=m.sha(m.EVAL),
                gap_core_sha256=m.sha(m.HERE/'gap_core.py'),
                gpu_end_to_end_rerun=False)
    if a.comparison:
        data=m.read(a.comparison)
        assert data['protocol']=='all_instances_class_agnostic_v1'
        rows=data['rows']; assert len(rows)==110
        expected={r['seq'] for r in m.read(m.HERE/'validation_pairs.json')['rows']}
        assert {r['scene'] for r in rows}==expected
        summary={}
        for method in ['baseline','current']:
            records=[]
            for row in rows:
                old=row[method]
                calc=m.composite({k:old[k] for k in ['AP','AP50','AP25']},old['D'])
                if old['GAP_v3'] is None: assert calc['GAP_v3'] is None
                else: assert abs(old['GAP_v3']-calc['GAP_v3'])<1e-12
                record_path=a.comparison.parent/'methods'/method/row['scene']/'metrics.json'
                record=m.read(record_path); d=record['deletion']
                den=d['TP']+d['FP']+d['FN']
                assert abs(old['D']-(d['TP']/den if den else 0.))<1e-12
                records.append(calc)
            summary[method]={k:float(np.mean([r[k] for r in records if r[k] is not None]))
                             for k in records[0]}
        report.update(historical_pairs_formula_checked=110,historical_method_records_checked=220,
                      comparison_sha256=m.sha(a.comparison),historical_scene_means=summary,
                      note='Formula/count verification of saved results only; no geometric AP rerun and no claim of new ablation results.')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    m.write(a.out,report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
