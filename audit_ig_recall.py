"""Summarize where IG visible-region detection loses candidate recall."""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path


def audit(root):
    root = Path(root)
    summary = json.loads((root / 'summary.json').read_text())
    refs = defaultdict(lambda: dict(frames=0, candidates=0, max_tested=0,
                                    max_conflict=0., max_observed=0, max_free=0,
                                    states=Counter()))
    reason = Counter()
    frames = 0
    for line in (root / 'frames.jsonl').read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line); frames += 1
        for item in row.get('references', []):
            key = str(item['reference_id']); out = refs[key]
            out['frames'] += 1
            out['candidates'] += int(bool(item.get('candidate')))
            out['max_tested'] = max(out['max_tested'], int(item.get('tested_reference', 0)))
            out['max_conflict'] = max(out['max_conflict'], float(item.get('reference_conflict_ratio', 0.)))
            out['max_observed'] = max(out['max_observed'], int(item.get('visible_cells', 0)))
            out['max_free'] = max(out['max_free'], int(item.get('confirmed_free_cells', 0)))
            out['states'][item.get('state', 'unknown')] += 1
            reason.update(item.get('failure_reasons', item.get('candidate_failures', [])))
    for value in refs.values():
        value['states'] = dict(value['states'])
    report = dict(detector_version=summary.get('detector_version'), frames=frames,
                  summary_audit_counts=summary.get('audit_counts', {}),
                  failure_reasons=dict(reason), references=dict(refs),
                  skips=summary.get('skips', {}))
    (root / 'recall_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ['detector_version', 'frames', 'failure_reasons']}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
