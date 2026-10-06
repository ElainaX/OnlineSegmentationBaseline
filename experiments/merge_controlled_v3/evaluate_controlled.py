"""Run unchanged AP formulas; actually write output (upstream CLI forgot it)."""
import json
from pathlib import Path
import sys

def dump_matches(matches, output):
    rows = []
    for scene, match in matches.items():
        for label, objects in match['gt'].items():
            for gt in objects:
                n = gt['vert_count']
                if gt['instance_id'] < 1000 or n < 100: continue
                candidates = []
                for pred in gt['matched_pred']:
                    intersection = pred['intersection']
                    candidates.append(dict(pred_id=pred['pred_id'],
                        iou=intersection/(n+pred['vert_count']-intersection),
                        coverage=intersection/n, purity=intersection/pred['vert_count'],
                        pred_points=pred['vert_count']))
                candidates.sort(key=lambda x:x['iou'], reverse=True)
                rows.append(dict(scene=scene, gt_id=gt['instance_id'], points=n,
                                 best=candidates[0] if candidates else None,
                                 candidates=candidates[:5]))
    Path(output+'.instances.json').write_text(json.dumps(rows,indent=2))

repo = Path('/root/OnlineSegmentationBaseline_baseline_ablation')
sys.path.insert(0, str(repo))
path = repo/'eval/evaluate_seqs.py'
src = path.read_text()
needle = 'evaluate(final_seq_name_list, pred_files, gt_files, recon_pc_list, gt_pc_list)'
assert src.count(needle) == 1
src = src.replace(needle, needle[:-1]+', output_file=opt.output_file)')
src = src.replace('    ap_scores = evaluate_matches(matches)',
                  '    dump_matches(matches, output_file)\n    ap_scores = evaluate_matches(matches)')
exec(compile(src, str(path), 'exec'), {'__name__':'__main__','__file__':str(path), 'dump_matches':dump_matches})
