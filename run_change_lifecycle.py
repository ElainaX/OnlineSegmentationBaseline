"""Opt-in independent-submap lifecycle runner. No GT-dependent decisions.

JSONL rows: {frame, packet, phase: reference|live}. Packet is an NPZ with rgb
(HxWx3 uint8), depth_m (HxW camera Z), K (3x3), pose (4x4 T_world_camera),
and mask__<provisional_fragment_key> boolean arrays. Reuse keys only within the
same temporal fragment, not frame-local mask ordinals. Reference packets come
first; live packets cannot write historical keys. See docs/change_lifecycle.md.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from submap_disjoint.independent_tsdf import IndependentTSDFStore, TSDFOptions
from submap_disjoint.change_lifecycle import ChangeLifecycle, LifecycleOptions
from submap_disjoint.contradiction import AuditOptions, validate_camera


def write_cloud(path, store):
    import open3d as o3d
    points, colors, _ = store.surface()
    if not len(points): return
    p = o3d.geometry.PointCloud()
    # This store integrates float RGB in [0,255]; its extracted colors retain
    # that range. Legacy Open3D PLY writers expect [0,1].
    p.points = o3d.utility.Vector3dVector(points)
    p.colors = o3d.utility.Vector3dVector(np.clip(colors / 255., 0., 1.))
    if not o3d.io.write_point_cloud(str(path), p): raise IOError('PLY export failed')


def demo(out, device):
    """Small geometry correctness scenario, not a ScanNet/3RScan benchmark."""
    K = np.array([[65., 0, 31.5], [0, 65., 23.5], [0, 0, 1]])
    rgb = np.full((48, 64, 3), 150, np.uint8)
    mask = np.zeros((48, 64), bool); mask[10:38, 15:49] = True
    cases = {}
    for case in ('static', 'occluded', 'disappeared_then_returned', 'new_occluder'):
        store = IndependentTSDFStore(TSDFOptions(device=device))
        store.create('old')
        for i in range(3): store.integrate('old', i, rgb, np.full(mask.shape, 2.013, np.float32), K, np.eye(4), mask)
        ctrl = ChangeLifecycle(store); ctrl.register_reference('old')
        folder = out / case; folder.mkdir()
        write_cloud(folder / 'before.ply', store)
        rows = []
        if case == 'new_occluder':
            sequence = [(1.013, 0., {'new': mask})] * 3 + [(1.013, x, {}) for x in (0., .12, -.12, .25, -.25)]
        elif case == 'disappeared_then_returned':
            sequence = [(z, x, {}) for z in (3.013, 2.013) for x in (0., .15, -.15)]
        else:
            sequence = [(2.013 if case == 'static' else 1.013, x, {}) for x in (0., .15, -.15)]
        for j, (z, x, assignments) in enumerate(sequence):
            pose = np.eye(4); pose[0, 3] = x
            rows.append(ctrl.step(j + 10, rgb, np.full(mask.shape, z, np.float32), K, pose, assignments))
        write_cloud(folder / 'after.ply', store)
        (folder / 'frames.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        cases[case] = dict(final_states={k: t.state for k, t in ctrl.tracks.items()}, events=ctrl.events,
            mean_step_seconds=float(np.mean([r['total_seconds'] for r in rows])))
    expected = {'static': {'old': 'stable'}, 'occluded': {'old': 'stable'},
                'disappeared_then_returned': {'old': 'stable'}, 'new_occluder': {'old': 'stable', 'new': 'accepted'}}
    for case, states in expected.items():
        if cases[case]['final_states'] != states: raise AssertionError((case, cases[case]))
    transitions = [e['after'] for e in cases['disappeared_then_returned']['events']]
    if 'retired' not in transitions: raise AssertionError('demo did not retire the disappeared surface')
    (out / 'summary.json').write_text(json.dumps(dict(kind='synthetic correctness, not AP', cases=cases), indent=2))
    print(json.dumps(cases, indent=2))


def run_manifest(args, out):
    config = json.loads(Path(args.config).read_text()) if args.config else {}
    if args.resume:
        if config: raise ValueError('resumed checkpoint retains its exact settings')
        ctrl = ChangeLifecycle.load(args.resume); store = ctrl.store; live_started = True
    else:
        store = IndependentTSDFStore(TSDFOptions(device=args.device, **config.get('tsdf', {})))
        audit = dict(near_m=store.options.near_m, far_m=store.options.far_m)
        audit.update(config.get('audit', {}))
        ctrl = ChangeLifecycle(store, LifecycleOptions(**config.get('lifecycle', {})), AuditOptions(**audit))
        live_started = False
    manifest = Path(args.manifest); previous = ctrl.last_frame
    log = out / 'frames.jsonl'
    with manifest.open() as source, log.open('w') as sink:
        for line in source:
            if not line.strip(): continue
            row = json.loads(line); frame = int(row['frame']); phase = row['phase']
            if previous is not None and frame <= previous: raise ValueError('manifest frames must increase')
            previous = frame
            path = Path(row['packet'])
            if not path.is_absolute(): path = manifest.parent / path
            with np.load(path, allow_pickle=False) as p:
                rgb, dep, K, pose = (p[k] for k in ('rgb', 'depth_m', 'K', 'pose'))
                masks = {k[6:]: p[k] for k in p.files if k.startswith('mask__')}
                validate_camera(K, pose)
                if phase == 'reference':
                    if live_started: raise ValueError('cannot modify historical reference after audit starts')
                    union = np.zeros(dep.shape, bool)
                    for key, mask in masks.items():
                        if mask.shape != dep.shape or np.any(union & mask): raise ValueError('reference masks must be disjoint')
                        union |= mask.astype(bool)
                    for key, mask in masks.items():
                        if key not in store.fragments:
                            if len(store.fragments) >= ctrl.options.max_fragments: raise MemoryError('fragment budget exceeded')
                            store.create(key)
                        store.integrate(key, frame, rgb, dep, K, pose, mask, {'packet': str(path)})
                    if sum(f.volume.bytes_allocated_attributes() for f in store.fragments.values()) > ctrl.options.max_attribute_bytes:
                        raise MemoryError('reference TSDF attribute budget exceeded')
                    result = dict(frame=frame, phase='reference')
                elif phase == 'live':
                    if not live_started:
                        for key in store.fragments: ctrl.register_reference(key)
                        live_started = True
                    result = ctrl.step(frame, rgb, dep, K, pose, masks)
                else: raise ValueError('phase must be reference or live')
            sink.write(json.dumps(result) + '\n'); sink.flush()
    if not live_started: raise ValueError('manifest contains no live audit frames')
    ctrl.save(out / 'checkpoint'); write_cloud(out / 'live_map.ply', store)
    (out / 'summary.json').write_text(json.dumps(dict(states={k: t.state for k, t in ctrl.tracks.items()},
        events=ctrl.events, ground_truth_used=False, probability_calibrated=False), indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--manifest'); mode.add_argument('--demo', action='store_true')
    ap.add_argument('--output', required=True); ap.add_argument('--device', default='CUDA:0')
    ap.add_argument('--resume'); ap.add_argument('--config')
    args = ap.parse_args(); out = Path(args.output)
    if out.exists(): raise FileExistsError('use a fresh output directory')
    out.mkdir(parents=True)
    if args.demo: demo(out, args.device)
    else: run_manifest(args, out)


if __name__ == '__main__': main()
