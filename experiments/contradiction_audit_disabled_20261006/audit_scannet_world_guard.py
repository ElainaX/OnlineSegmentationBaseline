#!/usr/bin/env python3
"""World-consistent contradiction guard for static/dynamic diagnostics.

A residual becomes a candidate at frame t from a free-space ray. The candidate
stores the old map surface point in world coordinates. Later audited views must
reproject to the same point, still see TSDF support for it, and observe a farther
surface. This is diagnostic-only; it never deletes instances.
"""
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np, open3d as o3d, open3d.core as o3c
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'experiments/contradiction_audit'))
from audit_3rscan import AuditDataset, raycast
from voxelized_points import VoxelBlockGrid
import tool.config as config


def edge_mask(depth_m, threshold):
    valid = depth_m > 0; edge = np.zeros_like(valid, dtype=bool)
    for axis in (0, 1):
        for shift in (-1, 1):
            sd = np.roll(depth_m, shift, axis=axis); sv = np.roll(valid, shift, axis=axis)
            edge |= valid & sv & (np.abs(depth_m - sd) > threshold)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    return edge


def candidate_points(depth_m, pred_m, free, edge, K, pose, stride, voxel):
    """Return one world point per quantized old-surface cell from sampled free rays."""
    h, w = depth_m.shape; s = max(1, int(stride)); yy, xx = np.mgrid[0:h:s, 0:w:s]
    f = free[::s, ::s] & ~edge[::s, ::s]
    z = pred_m[::s, ::s]; good = f & np.isfinite(z) & (z > 0)
    if not np.any(good): return {}
    x = (xx.astype(np.float32) - K[0, 2]) * z / K[0, 0]
    y = (yy.astype(np.float32) - K[1, 2]) * z / K[1, 1]
    cam = np.stack([x, y, z], axis=-1)[good]
    world = cam @ pose[:3, :3].T + pose[:3, 3]
    q = np.floor(world / float(voxel) + .5).astype(np.int32)
    out = {}
    for k, p in zip(map(tuple, q), world):
        if k not in out: out[k] = p.astype(np.float32)
    return out


def evaluate_active(active, depth_m, pred_m, K, pose, tau_min, tau_rel, map_gate):
    """Project active world candidates and count visible contradictions vectorially."""
    if not active: return 0, 0, 0
    keys = list(active); world = np.asarray([active[k]['point'] for k in keys], dtype=np.float32)
    w2c = np.linalg.inv(pose)
    cam = world @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]; h, w = depth_m.shape
    valid_z = np.isfinite(z) & (z > .1)
    u = np.full((len(z),), -1, dtype=np.int32)
    v = np.full((len(z),), -1, dtype=np.int32)
    u[valid_z] = np.rint(K[0, 0] * cam[valid_z, 0] / z[valid_z] + K[0, 2]).astype(np.int32)
    v[valid_z] = np.rint(K[1, 1] * cam[valid_z, 1] / z[valid_z] + K[1, 2]).astype(np.int32)
    inside = valid_z & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    idx = np.flatnonzero(inside)
    if idx.size == 0: return 0, 0, 0
    uz, vz = u[idx], v[idx]; obs = depth_m[vz, uz]; pred = pred_m[vz, uz]
    tau = np.maximum(tau_min, tau_rel * np.maximum(pred, .1))
    visible = (obs > 0) & (pred > 0) & (np.abs(pred - z[idx]) <= np.maximum(map_gate, .08 * z[idx]))
    contra = visible & (obs > z[idx] + tau)
    for j, is_contra in zip(idx[visible], contra[visible]):
        rec = active[keys[int(j)]]; rec['views'] += 1; rec['hits'] += int(is_contra); rec['last_seen'] = rec['audit']
    return int(visible.sum()), int(contra.sum()), int(sum(rec['hits'] >= 2 for rec in active.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True); ap.add_argument('--seq', required=True); ap.add_argument('--output-root', required=True)
    ap.add_argument('--config', default=str(REPO / 'config/scannet_stride10_controlled.yaml'))
    ap.add_argument('--audit-interval', type=int, default=10); ap.add_argument('--tau-min', type=float, default=.035); ap.add_argument('--tau-rel', type=float, default=.04)
    ap.add_argument('--ratio-threshold', type=float, default=.05); ap.add_argument('--min-pixels', type=int, default=400); ap.add_argument('--edge-threshold', type=float, default=.08)
    ap.add_argument('--candidate-stride', type=int, default=8); ap.add_argument('--candidate-voxel', type=float, default=.05); ap.add_argument('--max-age', type=int, default=8)
    ap.add_argument('--map-gate', type=float, default=.08); ap.add_argument('--min-hits', type=int, default=2); ap.add_argument('--max-candidates', type=int, default=150000)
    args = ap.parse_args()
    ds = AuditDataset(Path(args.data_root) / args.seq); cfg = config.load_config(args.config)
    wrap = VoxelBlockGrid(cfg, ds, 'cuda:0'); grid = wrap.voxel_block_grids; K_o3c = o3c.Tensor.from_numpy(ds.K.astype(np.float64)); dev = o3c.Device('CUDA:0')
    scale = float(cfg['cam']['depth_scale']); far = float(cfg['cam']['depth_far']); trunc = float(cfg['scene'].get('trunc_voxel_multiplier', 4.))
    out = Path(args.output_root) / args.seq; out.mkdir(parents=True, exist_ok=True); frames=[]; active={}; audit_no=0; last_pose=np.eye(4,dtype=np.float32)
    for n, fid in enumerate(ds.ids):
        depth_raw = cv2.imread(str(ds.depths[fid]), cv2.IMREAD_UNCHANGED); color = cv2.imread(str(ds.colors[fid]), cv2.IMREAD_COLOR); pose = ds.poses[fid]
        if not np.isfinite(pose).all(): pose=last_pose
        else: last_pose=pose
        pw = np.linalg.inv(pose); pre = None
        if n % max(1,args.audit_interval)==0:
            audit_no += 1; pre,_=raycast(grid,K_o3c,pw,depth_raw,ds.target_w,ds.target_h,scale,far,trunc)
            obs=depth_raw.astype(np.float32)/scale; both=(obs>0)&(pre>0); tau=np.maximum(args.tau_min,args.tau_rel*np.maximum(pre,.1)); free=both&(obs>pre+tau); edges=edge_mask(obs,args.edge_threshold)
            visible, repeated, confirmed_before = evaluate_active(active,obs,pre,ds.K,pose,args.tau_min,args.tau_rel,args.map_gate)
            # Age only candidates that were not visible; no visibility is not evidence for either side.
            for rec in active.values(): rec['audit']=audit_no
            stale=[k for k,r in active.items() if audit_no-r['last_added']>args.max_age and r['hits']<args.min_hits]
            for k in stale: del active[k]
            new=candidate_points(obs,pre,free,edges,ds.K,pose,args.candidate_stride,args.candidate_voxel)
            for k,p in new.items():
                if k not in active: active[k]={'point':p,'hits':1,'views':1,'last_added':audit_no,'last_seen':audit_no,'audit':audit_no}
            if len(active)>args.max_candidates:
                order=sorted(active,key=lambda k:(active[k]['hits']>=args.min_hits,active[k]['last_seen']),reverse=True)
                active={k:active[k] for k in order[:args.max_candidates]}
            confirmed=sum(r['hits']>=args.min_hits for r in active.values())
            frames.append({'frame':int(fid),'both_pixels':int(both.sum()),'free_pixels':int(free.sum()),'free_ratio':float(free.sum()/max(1,both.sum())),
                           'active_candidates':int(len(active)),'new_candidates':int(len(new)),'visible_prior':visible,'repeated_contradictions':repeated,
                           'confirmed_candidates':int(confirmed),'raw_trigger':bool(free.sum()>=args.min_pixels and free.sum()/max(1,both.sum())>=args.ratio_threshold),
                           'map_blocks':int(grid.hashmap().size())})
        rgb=cv2.cvtColor(color,cv2.COLOR_BGR2RGB); di=o3d.t.geometry.Image(np.ascontiguousarray(depth_raw)).to(o3c.uint16).to(dev); ci=o3d.t.geometry.Image(np.ascontiguousarray(rgb)).to(o3c.uint8).to(dev); ext=o3c.Tensor.from_numpy(np.ascontiguousarray(pw.astype(np.float64))); fc=grid.compute_unique_block_coordinates(di,K_o3c,ext,scale,far); grid.integrate(fc,di,ci,K_o3c,ext,scale,far,trunc_voxel_multiplier=trunc)
        if n%250==0: print(f'{args.seq}: {n+1}/{len(ds.ids)} blocks={grid.hashmap().size()} active={len(active)}',flush=True)
    ratio=np.asarray([x['free_ratio'] for x in frames],dtype=float); rep=np.asarray([x['repeated_contradictions'] for x in frames],dtype=float)
    summary={'sequence':args.seq,'frames_total':len(ds.ids),'audited_frames':len(frames),'parameters':vars(args),'meaning':{'raw_trigger':'single-view free-space residual','repeated_contradiction':'a previous old-map world point is visible again and current depth is farther while raycast still supports the old point','confirmed_candidate':'candidate with min-hits repeated contradictions; diagnostic only'},'metrics':{'raw_trigger_frames':int(sum(x['raw_trigger'] for x in frames)),'raw_trigger_rate':float(sum(x['raw_trigger'] for x in frames)/max(1,len(frames))),'mean_free_ratio':float(ratio.mean()) if len(ratio) else 0.,'total_repeated_contradictions':int(rep.sum()),'max_confirmed_candidates':int(max([x['confirmed_candidates'] for x in frames],default=0)),'max_active_candidates':int(max([x['active_candidates'] for x in frames],default=0))},'frames':frames}
    json.dump(summary,open(out/'world_guard_summary.json','w'),indent=2); print(json.dumps(summary['metrics'],indent=2),flush=True)
if __name__=='__main__': main()