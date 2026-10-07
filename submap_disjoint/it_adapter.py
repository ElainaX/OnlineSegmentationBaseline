"""Causal IG-disjoint -> independent Instance-TSDF fragments -> AP adapter.

Association remains the unchanged IG owner. RGB-D is buffered for one existing
merge period, grouped using ONLY the identities available at that period's end,
then fused into immutable, independently restorable TSDFs. Later identity merges
change references, never TSDF values. No correction or GT input is used here.
"""
import gc
import os
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import open3d as o3d
import open3d.core as o3c

from .independent_tsdf import IndependentTSDFStore, TSDFOptions
from tool.post_process import filter_instances, export_instance_mask


def group_observations(packets, global_to_merged):
    """Keep removed observations separate; -1 is NOT one shared identity."""
    groups = defaultdict(list)
    for pi, packet in enumerate(packets):
        groups[('background',)].append((pi, None, None))
        for raw_id, global_id in packet['pairs']:
            mid = int(global_to_merged[global_id])
            key = ('instance', mid) if mid >= 0 else ('orphan', global_id)
            groups[key].append((pi, raw_id, global_id))
    return groups


def resolve_fragment(global_ids, global_to_merged):
    mids = {int(global_to_merged[g]) for g in global_ids}
    if not mids or mids == {-1}: return -1
    if len(mids) != 1 or -1 in mids:
        raise RuntimeError('An immutable fragment was split by the association owner')
    return mids.pop()


def coalesce_surfaces(xyz, rgb, voxel_m):
    """Common world lattice for AP point ordering, not TSDF fusion.

One exported point per world voxel; preserve a source->point inverse map so
different instance supports can overlap exactly as in the IG exporter.
The original fragment volumes and surface files are never modified.
"""
    if not len(xyz): raise RuntimeError('Independent TSDFs produced no surface')
    keys = np.floor(xyz / voxel_m).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    count = np.bincount(inverse)
    p = np.stack([np.bincount(inverse, weights=xyz[:,i]) / count for i in range(3)], axis=1)
    c = np.stack([np.bincount(inverse, weights=rgb[:,i]) / count for i in range(3)], axis=1)
    return p, c, inverse


def reserve_window(volume, views, K):
    """Reserve the complete causal window before integrating any values.

    The bundled Open3D CUDA VBG can invalidate attribute pointers when growing
    an already populated hashmap. Precompute keys, not geometry, to prevent
    mid-integration reallocation. This does not change TSDF samples or weights.
    """
    keys=[]; opt=volume.options; intrinsic=o3c.Tensor(np.asarray(K,dtype=np.float64))
    for packet,mask in views:
        dep=packet['depth']; valid=np.isfinite(dep)&(dep>opt.near_m)&(dep<opt.far_m)&mask
        if not valid.any():continue
        image=o3d.t.geometry.Image(np.ascontiguousarray(np.where(valid,dep,0),dtype=np.float32)).to(volume.device)
        extrinsic=o3c.Tensor(np.linalg.inv(packet['pose'])@volume.anchor)
        try:
            k=volume.grid.compute_unique_block_coordinates(image,intrinsic,extrinsic,1.,opt.far_m,
                                                            trunc_voxel_multiplier=opt.trunc_voxels)
        except RuntimeError as e:
            if 'No block is touched' in str(e):continue
            raise
        keys.append(k.cpu().numpy())
    needed=len(np.unique(np.concatenate(keys),axis=0)) if keys else 0
    if needed>opt.max_blocks:raise MemoryError('Independent fragment exceeds block budget: '+str(needed))
    if needed*4>volume.grid.hashmap().capacity():
        volume.grid.hashmap().reserve(needed*4+1)


class InstanceTSDFAdapter:
    def __init__(self, cfg, owner, directory):
        self.owner = owner
        self.cfg = cfg
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.options = TSDFOptions(
            voxel_m=float(cfg['scene']['voxel_size']),
            block_resolution=int(cfg['scene']['block_resolution']),
            trunc_voxels=float(cfg['scene'].get('trunc_voxel_multiplier',4.)),
            near_m=float(cfg['cam']['depth_near']), far_m=float(cfg['cam']['depth_far']),
            max_blocks=int(cfg['scene']['block_count']), device=owner.device.upper(),
            # Open3D uses weight > threshold; 0.5 retains >=1 observation.
            extraction_weight=.5)
        self.K = owner.intrinsics.detach().cpu().numpy().astype(np.float64)
        self.packets = []
        self.fragments = []
        self.windows = []
        self.audit = dict(buffer_seconds=0., local_tsdf_seconds=0., persistence_seconds=0.,
                          export_seconds=0., buffered_frames_peak=0, windows=0,
                          fragments=0, accepted_masks=0, integrated_fragment_frames=0,
                          raw_surface_points=0, peak_volume_attribute_bytes=0,
                          volume_blocks_total=0)

    def observe(self, frame_id, rgb, depth, pose, seg, pairs):
        started = time.perf_counter()
        rgb = (rgb.detach().cpu().numpy()*255.).astype(np.uint8)
        depth = depth.detach().cpu().numpy().astype(np.float32)
        pose = pose.detach().cpu().numpy().astype(np.float64)
        labels = seg.detach().cpu().numpy().astype(np.int32) if seg is not None else np.zeros(depth.shape,np.int32)
        accepted = [(int(p['mask_id']),int(p['global_mask_id'])) for p in pairs]
        self.packets.append(dict(frame=int(frame_id),rgb=rgb,depth=depth,pose=pose,labels=labels,pairs=accepted))
        self.audit['accepted_masks'] += len(accepted)
        self.audit['buffered_frames_peak'] = max(self.audit['buffered_frames_peak'],len(self.packets))
        self.audit['buffer_seconds'] += time.perf_counter()-started
        if frame_id > 0 and frame_id % self.owner.merge_frame_interval == 0:
            self.flush(frame_id)

    def flush(self, decision_frame):
        if not self.packets: return
        debug = bool(os.environ.get('IT_TRACE'))
        if debug: print('IT_FLUSH_BEGIN',decision_frame,flush=True)
        groups = group_observations(self.packets,self.owner.global2merged_mask_id)
        window = dict(decision_frame=int(decision_frame),frames=[p['frame'] for p in self.packets],fragments=[])
        assert max(window['frames']) <= decision_frame, 'future information leaked into fragment formation'
        # Each frame is partitioned exactly once: accepted masks and residual.
        residual = [~np.isin(p['labels'],[x[0] for x in p['pairs']]) for p in self.packets]
        for key, observations in groups.items():
            if debug: print('IT_CREATE',key,flush=True)
            fid = 'fragment_%06d' % len(self.fragments)
            target = self.directory/fid
            store = IndependentTSDFStore(self.options)
            store.create(fid)  # common world anchor preserves the voxel lattice
            grouped = defaultdict(list)
            for pi,raw,gid in observations: grouped[pi].append((raw,gid))
            t = time.perf_counter()
            views=[]
            for pi,members in grouped.items():
                p=self.packets[pi]
                mask=residual[pi] if key[0]=='background' else np.isin(p['labels'],[x[0] for x in members])
                views.append((p,mask))
            reserve_window(store.fragments[fid].volume,views,self.K)
            for (pi, members),(p,mask) in zip(grouped.items(),views):
                if debug: print('IT_INTEGRATE',key,pi,flush=True)
                store.integrate(fid,p['frame'],p['rgb'],p['depth'],self.K,p['pose'],mask,
                                source={'raw_mask_ids':[x[0] for x in members if x[0] is not None]})
            volume = store.fragments[fid].volume
            if debug: print('IT_SURFACE',key,flush=True)
            xyz, colors = volume.surface()
            o3c.cuda.synchronize()
            self.audit['local_tsdf_seconds'] += time.perf_counter()-t
            self.audit['peak_volume_attribute_bytes'] = max(self.audit['peak_volume_attribute_bytes'],volume.bytes_allocated_attributes())
            blocks = int(volume.grid.hashmap().size())
            self.audit['volume_blocks_total'] += blocks
            self.audit['integrated_fragment_frames'] += volume.integrations
            self.audit['raw_surface_points'] += len(xyz)
            gids = sorted({gid for _,_,gid in observations if gid is not None})
            store.link('window_identity_'+str(key),[fid])
            store.set_status(fid,'frozen','End of existing IG merge period; no correction applied')
            t = time.perf_counter()
            store.save(target,save_volumes=True)
            np.savez_compressed(target/'surface.npz',points=xyz.astype(np.float32),colors=colors.astype(np.float32))
            self.audit['persistence_seconds'] += time.perf_counter()-t
            record = dict(key=fid,kind=key[0],window=len(self.windows),global_ids=gids,
                          observation_frames=sorted({self.packets[pi]['frame'] for pi,_,_ in observations}),
                          surface_points=len(xyz),blocks=blocks)
            self.fragments.append(record); window['fragments'].append(fid)
            del volume, store
        self.windows.append(window)
        self.packets.clear()
        self.audit.update(windows=len(self.windows),fragments=len(self.fragments))
        print('IT_WINDOW',json.dumps(dict(frame=decision_frame,fragments=len(window['fragments']),
              total_fragments=len(self.fragments),local_seconds=self.audit['local_tsdf_seconds'])),flush=True)

    def export(self, frame_id, output):
        self.flush(frame_id)
        started = time.perf_counter(); output=Path(output)
        points=[]; colors=[]; lengths=[]; assignments=[]
        for f in self.fragments:
            with np.load(self.directory/f['key']/'surface.npz') as surface:
                points.append(surface['points']); colors.append(surface['colors']); lengths.append(len(surface['points']))
            assignments.append(resolve_fragment(f['global_ids'],self.owner.global2merged_mask_id))
        points,colors,inverse = coalesce_surfaces(np.concatenate(points),np.concatenate(colors),self.options.voxel_m)
        support=defaultdict(list); offset=0
        for mid,n in zip(assignments,lengths):
            if mid>=0: support[mid].append(inverse[offset:offset+n])
            offset+=n
        threshold=min(self.owner.merge_time+2,self.cfg['seg']['mask_weight_threshold'])
        valid=[i for i in range(self.owner.c_mask_num) if float(self.owner.merged_mask_weight[i])>=threshold]
        masks=[]; features=[]; ids=[]
        for mid in valid:
            indices=np.unique(np.concatenate(support[mid])) if mid in support else np.empty(0,np.int64)
            if len(indices)<25: continue
            mask=np.zeros(len(points),bool); mask[indices]=True
            masks.append(mask); ids.append(mid)
            features.append(self.owner.maskGraph.instance_dict[mid].get_semantic_feature.detach().cpu().numpy())
        # The existing exporter sorts masks by ascending size in return_list mode
        # and retains overlapping masks. Preserve that behavior for class-agnostic
        # AP; keep metadata/features in the SAME order (no semantic relabeling).
        if masks:
            order=torch.argsort(torch.tensor([int(m.sum()) for m in masks],device=self.owner.device)).cpu().tolist()
            masks=[masks[i] for i in order]; features=[features[i] for i in order]; ids=[ids[i] for i in order]
            masks,features,kept=filter_instances(points,masks,features)
            ids=[ids[i] for i in kept]
        self.owner.save_pc(points,colors/255.,str(output/'final.ply'))
        export_instance_mask(str(output/'ckpt_final.npz'),masks,features)
        if not masks:
            np.savez(output/'ckpt_final.npz',pred_masks=np.zeros((len(points),0),bool),pred_score=np.ones(0),
                     pred_classes=np.zeros(0,np.int32),pred_sem_features=np.zeros((self.cfg['mask']['feature_dim'],0)))
        self.audit.update(export_seconds=time.perf_counter()-started,export_points=len(points),export_instances=len(masks))
        identities=defaultdict(list)
        for f,mid in zip(self.fragments,assignments):
            if mid>=0: identities[str(mid)].append(f['key'])
        report=dict(options=vars(self.options),association='unchanged IG sparse disjoint',
                    fragment_policy='one existing merge period, causal identities at period end',
                    correction_enabled=False,carving_aabb_enabled=False,depth_shrink_enabled=False,
                    identity_links_fuse_geometry=False,geometry_fallback=False,
                    association_geometry='shared TSDF used for original overlap and FCGF only',
                    export_geometry='independent TSDF surfaces; world-voxel coalescing for point indexing',
                    extraction_weight_note='IT strict threshold=0.5 retains >=1 observation; shared Open3D default=3; weights are not pooled across volumes',
                    fragments=self.fragments,windows=self.windows,identities=dict(identities),
                    exported_instance_ids=ids,audit=self.audit)
        (output/'it_manifest.json').write_text(json.dumps(report,indent=2))
        return report
