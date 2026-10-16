"""AP matching, shared GAP-v3 domains, deletion audit, and pooled PR aggregation.

All numeric scores are fractions, not percentages. See docs/CURRENT_SYSTEM.md.
The CPU AP reducer executes the original evaluator functions, not a new AP rule.
"""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import runpy
import sys
from types import SimpleNamespace
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EVAL = ROOT / 'eval/evaluate_seqs.py'
PROTOCOL = '3rscan-all-positive-objectId-class-agnostic-gap-v3-incident-edges-v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write(path, value):
    with Path(path).open('w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def reducer():
    """Load only original NumPy AP functions, avoiding CUDA imports for pooling."""
    tree = ast.parse(EVAL.read_text(encoding='utf-8'))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('evaluate_matches', 'compute_averages')]
    assert len(functions) == 2
    constants = runpy.run_path(str(ROOT / 'eval/constants.py'))
    env = dict(np=np, deepcopy=deepcopy, CLASS_LABELS=constants['SCANNET_LABELS'],
               opt=SimpleNamespace(overlaps=np.append(np.arange(.5, .95, .05), .25),
                                   min_region_sizes=np.array([100]),
                                   distance_threshes=np.array([np.inf]),
                                   distance_confs=np.array([-np.inf])))
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(EVAL), 'exec'), env)
    return env


def ap_from_matches(matches):
    env = reducer()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # Undefined classes stay NaN/null.
        scores = env['evaluate_matches'](matches)
        avg = env['compute_averages'](scores)
    return clean(dict(AP=avg['all_ap'], AP50=avg['all_ap_50%'], AP25=avg['all_ap_25%']))


def composite(ap, d):
    assert np.isfinite(d) and 0 <= d <= 1
    result = dict(ap, D=float(d))
    for source, name in [('AP', 'GAP_v3'), ('AP50', 'GAP50'), ('AP25', 'GAP25')]:
        v = ap[source]
        if v is not None:
            assert np.isfinite(v) and 0 <= v <= 1
        result[name] = None if v is None else float(v + (1-v)*d)
    return result


def new_output(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def select_pair(a):
    rows = read(HERE / 'validation_pairs.json')['rows']
    selected = [r for r in rows if r['seq'] == a.seq]
    if len(selected) != 1:
        raise ValueError(f'Expected one known pair, found {len(selected)} for {a.seq}')
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    write(a.out, {'selected': selected})


def build_domain(a):
    import gap_core as g
    out = new_output(a.out)
    b = read(a.boundary)
    if b['sequence_name'] != a.seq:
        raise ValueError('Boundary scene mismatch')
    cut = int(b['rescan_start_frame'])
    xyz, labels0 = g.reference(a.reference_ply)
    ref, keys, labels = g.atoms(xyz, labels0)
    removed = set(map(int, b['changed_instances']['removed']))
    moved = {int(x['instance_reference']) for x in b['changed_instances']['rigid']} - removed
    eligible = np.isin(labels, list(removed | moved))
    occupied = g.nearest(ref, g.cloud(a.gt_ply))[0] <= g.TOL
    visible, free = g.observations(ref, a.frames, cut)
    target = eligible & ~occupied & visible & free
    neutral = eligible & ~occupied & ~target
    np.savez_compressed(out / 'domain.npz', points=ref, keys=keys, labels=labels,
                        target=target, visible=visible, free=free, occupied=occupied, neutral=neutral)
    frame_files = sorted(a.frames.glob('pose/*.txt')) + sorted(a.frames.glob('depth/*.png'))
    frame_files += [a.frames / 'intrinsic/intrinsic_depth.txt']
    inputs = [a.boundary, a.reference_ply, a.gt_ply] + frame_files
    write(out / 'domain.json', dict(protocol=PROTOCOL, seq=a.seq, cut=cut,
          removed_ids=sorted(removed), moved_ids=sorted(moved), target_atoms=int(target.sum()),
          reference_atoms=len(ref), neutral_atoms=int(neutral.sum()),
          domain_sha256=sha(out / 'domain.npz'), gap_core_sha256=sha(HERE / 'gap_core.py'),
          gt_ply_sha256=sha(a.gt_ply), inputs_sha256={str(p.resolve()): sha(p) for p in inputs}))


def geometric_matches(a):
    sys.path.insert(0, str(ROOT))
    old = sys.argv
    try:
        sys.argv = [str(EVAL), '--result_dir', str(a.result), '--gt_dir', str(a.gt_ply.parent),
                    '--output_file', str(a.out / 'unused.txt'), '--no_class']
        env = runpy.run_path(str(EVAL), run_name='current_system_ap_library')
    finally:
        sys.argv = old
    import open3d as o3d
    pred = a.result / 'ckpt_final.npz'
    gt_labels = np.loadtxt(a.gt_labels).reshape(-1)
    if not np.isfinite(gt_labels).all() or not np.equal(gt_labels, np.floor(gt_labels)).all():
        raise ValueError('GT labels must be finite integers')
    if not np.all((gt_labels == 0) | ((gt_labels > 3000) & (gt_labels < 4000))):
        raise ValueError('Expected all-instance encoding: 0 or 3000 + positive objectId < 1000')
    gt_pc = o3d.io.read_point_cloud(str(a.gt_ply))
    pc = o3d.io.read_point_cloud(str(a.result / 'final.ply'))
    if len(gt_pc.points) != len(gt_labels):
        raise ValueError('GT PLY vertex order/count must match GT labels')
    with np.load(pred) as z:
        if z['pred_masks'].shape != (len(pc.points), len(z['pred_score'])):
            raise ValueError('Prediction mask axes must be [final.ply points, instances]')
        if len(z['pred_score']) == 0:
            raise ValueError('Original geometric matcher cannot handle zero prediction columns; record failure, do not drop scene or invent zero')
    gt, pr = env['assign_instances_for_scan'](a.seq, str(pred), str(a.gt_labels), pc, gt_pc,
                                             save_seg_gt=False)
    return {a.seq: {'gt': gt, 'pred': pr}}


def evaluate(a):
    import gap_core as g
    meta = read(a.domain / 'domain.json')
    if meta['seq'] != a.seq or meta['protocol'] != PROTOCOL:
        raise ValueError('Domain scene/protocol mismatch')
    if meta['domain_sha256'] != sha(a.domain / 'domain.npz') or meta['gt_ply_sha256'] != sha(a.gt_ply):
        raise ValueError('Domain or aligned GT changed')
    if meta['gap_core_sha256'] != sha(HERE / 'gap_core.py'):
        raise ValueError('Domain generated with a different GAP implementation')
    summary = a.result / 'correction/summary.json'
    if not summary.exists() and not a.no_actions:
        raise FileNotFoundError('Missing correction/summary.json; use --no-actions only for a declared no-erasure run')
    if a.no_actions and summary.exists() and read(summary).get('actions'):
        raise ValueError('--no-actions conflicts with recorded actions')
    if summary.exists():
        for action in read(summary)['actions']:
            if not action.get('archive') and action.get('actual_erased_voxels') != 0:
                raise ValueError('Action lacks archive; cannot certify D')
    a.out = new_output(a.out)
    matches = geometric_matches(a)
    ap = ap_from_matches(matches)
    with np.load(a.domain / 'domain.npz') as z:
        d = g.evaluate_method(a.result, z['points'], z['keys'], z['labels'], z['target'],
                              set(meta['removed_ids']), set(meta['moved_ids']), z['neutral'],
                              meta['cut'], g.cloud(a.result / 'final.ply'),
                              archive_root=a.result / 'correction')
    files = [a.result / 'final.ply', a.result / 'ckpt_final.npz', a.gt_ply, a.gt_labels]
    if summary.exists():
        files.append(summary)
    record = dict(protocol=PROTOCOL, seq=a.seq, method=a.method, metrics=composite(ap, d['D']),
                  deletion=d, domain_sha256=meta['domain_sha256'],
                  ap_evaluator_sha256=sha(EVAL), gap_core_sha256=sha(HERE / 'gap_core.py'),
                  no_actions_declared=a.no_actions, inputs_sha256={str(p): sha(p) for p in files})
    # Pickle is a trusted local intermediate only; never load untrusted pickles.
    with (a.out / 'ap_matches.pkl').open('wb') as f:
        pickle.dump(matches, f, protocol=4)
    record['matches_sha256'] = sha(a.out / 'ap_matches.pkl')
    write(a.out / 'metrics.json', record)
    print(json.dumps(clean(record['metrics'])))


def pool_directories(directories):
    records, matches, method = [], {}, None
    for directory in directories:
        directory = Path(directory)
        r = read(directory / 'metrics.json')
        assert r['protocol'] == PROTOCOL and r['ap_evaluator_sha256'] == sha(EVAL)
        assert r['gap_core_sha256'] == sha(HERE / 'gap_core.py')
        assert r['matches_sha256'] == sha(directory / 'ap_matches.pkl')
        if method is None:
            method = r['method']
        assert r['method'] == method, 'Pool exactly one method at a time'
        assert r['seq'] not in matches, 'Duplicate scene'
        with (directory / 'ap_matches.pkl').open('rb') as f:
            m = pickle.load(f)
        assert list(m) == [r['seq']]
        matches.update(m)
        records.append(r)
    if not records:
        raise ValueError('No completed evaluation directories')
    fields = ['AP', 'AP50', 'AP25', 'D', 'GAP_v3', 'GAP50', 'GAP25']
    means, counts = {}, {}
    for k in fields:
        values = [r['metrics'][k] for r in records if r['metrics'][k] is not None]
        means[k] = float(np.mean(values)) if values else None
        counts[k] = len(values)
    for k in ['precision', 'recall']:
        values = [r['deletion'][k] for r in records if r['deletion'][k] is not None]
        means['deletion_' + k] = float(np.mean(values)) if values else None
        counts['deletion_' + k] = len(values)
    return dict(protocol=PROTOCOL, method=method, completed=len(records),
                scenes=[r['seq'] for r in records], pooled_AP=ap_from_matches(matches),
                scene_macro_mean=means, valid_scene_counts=counts,
                domains={r['seq']: r['domain_sha256'] for r in records},
                note='Pooled AP merges PR matching records. GAP/D are scene means; no pooled GAP is defined.')


def pooled(a):
    paths = read(a.manifest)
    if not isinstance(paths, list):
        raise ValueError('Manifest must be an explicit JSON list of evaluation directories')
    directories = [(a.manifest.parent / Path(p)).resolve() for p in paths]
    out = new_output(a.out)
    report = pool_directories(directories)
    write(out / 'pooled.json', report)
    print(json.dumps(clean(report)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('select-pair', help='Select one fixed validation pair')
    s.add_argument('--seq', required=True); s.add_argument('--out', type=Path, required=True)
    s.set_defaults(func=select_pair)
    s = sub.add_parser('domain', help='Build one method-independent GAP domain (CPU)')
    for k in ['boundary', 'reference-ply', 'gt-ply', 'frames', 'out']:
        s.add_argument('--' + k, type=Path, required=True)
    s.add_argument('--seq', required=True); s.set_defaults(func=build_domain)
    s = sub.add_parser('evaluate', help='AP matching (CUDA) and archive-supported GAP-v3 (CPU)')
    for k in ['result', 'gt-ply', 'gt-labels', 'domain', 'out']:
        s.add_argument('--' + k, type=Path, required=True)
    s.add_argument('--seq', required=True); s.add_argument('--method', required=True)
    s.add_argument('--no-actions', action='store_true')
    s.set_defaults(func=evaluate)
    s = sub.add_parser('pooled', help='Merge trusted saved matching records for ONE method (CPU)')
    s.add_argument('--manifest', type=Path, required=True)
    s.add_argument('--out', type=Path, required=True); s.set_defaults(func=pooled)
    a = p.parse_args()
    for k, v in vars(a).items():
        if isinstance(v, Path):
            setattr(a, k, v.resolve())
    a.func(a)


if __name__ == '__main__':
    main()
