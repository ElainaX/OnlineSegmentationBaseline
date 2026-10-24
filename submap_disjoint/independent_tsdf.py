"""Independent geometric state, deliberately separate from instance identity.

Units: meters. Poses: T_world_camera and T_world_submap. Depth: camera Z.
Frozen fragments remain in the current map; inactive fragments do not.
Identity links NEVER fuse TSDF values. No automatic change detector is implied.
"""
from dataclasses import dataclass, field
from typing import Dict
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d
import open3d.core as o3c


@dataclass(frozen=True)
class TSDFOptions:
    voxel_m: float = .025
    trunc_voxels: float = 4.
    block_resolution: int = 8
    initial_blocks: int = 256
    max_blocks: int = 30000
    near_m: float = .1
    far_m: float = 5.
    extraction_weight: float = 1.
    device: str = 'CUDA:0'

    def __post_init__(self):
        if not (self.voxel_m > 0 and self.trunc_voxels > 0 and
                0 <= self.near_m < self.far_m and
                0 < self.initial_blocks <= self.max_blocks):
            raise ValueError('invalid TSDF options')


class LocalTSDF:
    def __init__(self, options=TSDFOptions(), T_world_submap=None):
        self.options = options
        self.anchor = np.eye(4) if T_world_submap is None else np.array(T_world_submap, dtype=float)
        if self.anchor.shape != (4,4) or not np.isfinite(self.anchor).all():
            raise ValueError('invalid submap anchor')
        if not (np.allclose(self.anchor[3], [0,0,0,1]) and
                np.allclose(self.anchor[:3,:3].T @ self.anchor[:3,:3], np.eye(3), atol=1e-5) and
                np.linalg.det(self.anchor[:3,:3]) > .999):
            raise ValueError('anchor must be a proper rigid transform')
        self.device = o3c.Device(options.device)
        self.grid = o3d.t.geometry.VoxelBlockGrid(
            ('tsdf','weight','color'), (o3c.float32,)*3, ((1,),(1,),(3,)),
            options.voxel_m, options.block_resolution, options.initial_blocks, device=self.device)
        self.integrations = 0

    def integrate(self, rgb, depth_m, K, T_world_camera, mask=None):
        depth_m = np.asarray(depth_m)
        valid = np.isfinite(depth_m) & (depth_m > self.options.near_m) & (depth_m < self.options.far_m)
        if mask is not None:
            if np.shape(mask) != depth_m.shape: raise ValueError('mask/depth shape mismatch')
            valid &= np.asarray(mask, bool)
        if not valid.any(): return 0
        if rgb.shape != depth_m.shape + (3,) or rgb.dtype != np.uint8:
            raise ValueError('RGB must be H x W x 3 uint8')
        if not np.isfinite(T_world_camera).all(): raise ValueError('nonfinite camera pose')
        # Float Z avoids uint16 rounding; zero means unknown, never free space.
        dep = o3d.t.geometry.Image(np.ascontiguousarray(np.where(valid, depth_m, 0), dtype=np.float32)).to(self.device)
        # Open3D's float-depth kernel also requires float color (0..255).
        color = o3d.t.geometry.Image(np.ascontiguousarray(rgb, dtype=np.float32)).to(self.device)
        intrinsic = o3c.Tensor(np.ascontiguousarray(K, dtype=np.float64))
        extrinsic = o3c.Tensor(np.linalg.inv(T_world_camera) @ self.anchor)
        try:
            blocks = self.grid.compute_unique_block_coordinates(
                dep, intrinsic, extrinsic, 1., self.options.far_m,
                trunc_voxel_multiplier=self.options.trunc_voxels)
        except RuntimeError as error:
            if 'No block is touched' in str(error): return 0
            raise
        if blocks.shape[0] == 0: return 0
        _, exists = self.grid.hashmap().find(blocks)
        needed = int(self.grid.hashmap().size()) + int((~exists.cpu().numpy()).sum())
        if needed > self.options.max_blocks:
            raise MemoryError('local TSDF needs %d blocks, limit %d; refusing silent clipping' %
                              (needed, self.options.max_blocks))
        # The bundled stdgpu hash backend can exhaust collision buckets well
        # before the nominal capacity. Reserve headroom BEFORE inserting and
        # check every requested key afterwards rather than accepting lost blocks.
        capacity=int(self.grid.hashmap().capacity())
        if needed*4 > capacity:
            self.grid.hashmap().reserve(max(needed*4,capacity*2))
        self.grid.integrate(blocks, dep, color, intrinsic, extrinsic, 1., self.options.far_m,
                            trunc_voxel_multiplier=self.options.trunc_voxels)
        _, present=self.grid.hashmap().find(blocks)
        if not present.cpu().numpy().all():
            raise RuntimeError('TSDF hash allocation dropped requested blocks')
        self.integrations += 1
        return int(valid.sum())

    def surface(self, weight_threshold=None):
        if not self.grid.hashmap().size():
            return np.empty((0,3)), np.empty((0,3))
        threshold = self.options.extraction_weight if weight_threshold is None else float(weight_threshold)
        if not np.isfinite(threshold) or threshold <= 0: raise ValueError('invalid extraction weight')
        pc = self.grid.extract_point_cloud(weight_threshold=threshold)
        xyz = pc.point.positions.cpu().numpy()
        return xyz @ self.anchor[:3,:3].T + self.anchor[:3,3], pc.point.colors.cpu().numpy()

    def _aabb_voxels(self, lower, upper):
        lower=np.asarray(lower); upper=np.asarray(upper)
        if lower.shape!=(3,) or upper.shape!=(3,) or not np.isfinite([lower,upper]).all() or (upper<lower).any():
            raise ValueError('invalid world AABB')
        corners=np.array([[x,y,z] for x in (lower[0],upper[0]) for y in (lower[1],upper[1]) for z in (lower[2],upper[2])])
        local=(corners-self.anchor[:3,3])@self.anchor[:3,:3]
        extent=self.options.voxel_m*self.options.block_resolution
        lo=np.floor(local.min(0)/extent).astype(int)-1
        hi=np.floor(local.max(0)/extent).astype(int)+1
        if np.prod(hi-lo+1)>2000000: raise MemoryError('AABB query exceeds block-query budget')
        keys=np.stack(np.meshgrid(*(np.arange(a,b+1) for a,b in zip(lo,hi)),indexing='ij'),axis=-1).reshape(-1,3).astype(np.int32)
        buf,exists=self.grid.hashmap().find(o3c.Tensor(keys,device=self.device))
        active=buf[exists]
        if not active.shape[0]: return np.empty((0,3)),np.empty(0,np.int64)
        coords,indices=self.grid.voxel_coordinates_and_flattened_indices(active)
        world=coords.cpu().numpy()@self.anchor[:3,:3].T+self.anchor[:3,3]
        return world,indices.cpu().numpy().reshape(-1)

    def _clear_indices(self, selected):
        if len(selected):
            idx = o3c.Tensor(selected, dtype=o3c.int64, device=self.device)
            self.grid.attribute('weight').reshape((-1,1))[idx] = 0.
            self.grid.attribute('tsdf').reshape((-1,1))[idx] = 0.
            self.grid.attribute('color').reshape((-1,3))[idx] = 0.
        return len(selected)

    def clear_aabb(self, lower, upper):
        """Global-TSDF control: invalidate all voxel contributions inside a world box.

        This is explicitly a region clear, not an exact per-observation inverse.
        It can also be applied repeatedly to a union of tight boxes.
        """
        if not self.grid.hashmap().size(): return 0
        world, indices = self._aabb_voxels(lower,upper)
        inside = ((world >= lower) & (world <= upper)).all(1)
        return self._clear_indices(indices[inside])

    def clear_observed_support(self, source):
        """Stronger global control: clear exactly voxels touched by old target.

        The source encodes oracle per-pixel contribution support. This still
        clears mixed background contributions in those same voxels. Unlike
        inverse integration, it does not subtract individual weighted samples.
        """
        if self.options.voxel_m!=source.options.voxel_m or not np.allclose(self.anchor,source.anchor):
            raise ValueError('support clear requires identical voxel lattice')
        active=source.grid.hashmap().active_buf_indices()
        if not active.shape[0]: return 0
        coords,indices=source.grid.voxel_coordinates_and_flattened_indices(active)
        weight=source.grid.attribute('weight').reshape((-1,1))[indices].cpu().numpy().reshape(-1)
        p=coords.cpu().numpy()[weight>0]@source.anchor[:3,:3].T+source.anchor[:3,3]
        if not len(p): return 0
        q,indices=self._aabb_voxels(p.min(0),p.max(0))
        if not len(q): return 0
        exact=cKDTree(p).query(q,workers=4)[0]<self.options.voxel_m*.001
        return self._clear_indices(indices[exact])

    def bytes_allocated_attributes(self):
        return sum(int(np.prod(self.grid.attribute(k).shape))*4 for k in ('tsdf','weight','color'))


@dataclass
class Fragment:
    key: str
    volume: LocalTSDF
    status: str = 'active'
    observations: list = field(default_factory=list)
    seen: set = field(default_factory=set)
    masks: dict = field(default_factory=dict)


class IndependentTSDFStore:
    """Owner of independent volumes. Callers supply local state assignments.

    A global ID is a grouping of fragment references, not a TSDF accumulator.
    No geometric state is deleted merely because two IDs are linked.
    Deactivation affects the live map, leaving history available for reactivation.
    """
    def __init__(self, options=TSDFOptions()):
        self.options = options
        self.fragments: Dict[str, Fragment] = {}
        self.identities = {}
        self.events = []

    def create(self, key, anchor=None, identity=None):
        key = str(key)
        if key in self.fragments: raise ValueError('fragment already exists: '+key)
        self.fragments[key] = Fragment(key, LocalTSDF(self.options, anchor))
        if identity is not None: self.link(identity, [key])
        self.events.append({'operation':'create','fragment':key})
        return key

    def integrate(self, key, frame, rgb, depth_m, K, pose, mask, source=None):
        f = self.fragments[str(key)]
        if f.status not in ('active', 'pending'): raise RuntimeError('cannot integrate into '+f.status+' fragment')
        # One union mask per fragment and frame. Different masks in the same
        # frame must be unioned by the caller before fusion to avoid overweighting.
        digest = hashlib.sha256(np.packbits(np.asarray(mask,bool)).tobytes()).hexdigest()
        if frame in f.seen:
            prev = next(x for x in f.observations if x['frame']==int(frame))
            if prev['mask_sha256'] != digest: raise ValueError('different mask for previously integrated frame')
            return False
        n = f.volume.integrate(rgb, depth_m, K, pose, mask)
        if not n: return False
        f.seen.add(frame)
        f.masks[int(frame)] = np.packbits(np.asarray(mask,bool))
        f.observations.append({'frame':int(frame), 'mask_sha256':digest,
                               'mask_shape':list(np.shape(mask)), 'K':np.asarray(K).tolist(),
                               'pixels':n,'source':source,'T_world_camera':np.asarray(pose).tolist()})
        return True

    def link(self, identity, keys):
        keys = {str(k) for k in keys}
        if not keys.issubset(self.fragments): raise KeyError('unknown fragment')
        self.identities.setdefault(str(identity), set()).update(keys)
        self.events.append({'operation':'link','identity':str(identity),'fragments':sorted(keys)})

    def set_status(self, key, status, reason):
        if status not in ('active','frozen','inactive','pending'): raise ValueError('invalid lifecycle state')
        if not reason: raise ValueError('record why a state changes')
        f=self.fragments[str(key)]
        self.events.append({'operation':'state','fragment':str(key),'before':f.status,
                            'after':status,'reason':reason})
        f.status=status

    def surface(self, include_inactive=False):
        points=[]; colors=[]; fragment_keys=[]
        for key,f in self.fragments.items():
            # Pending geometry is queryable for audits, never published early.
            if f.status in ('inactive', 'pending') and not include_inactive: continue
            p,c=f.volume.surface()
            points.append(p); colors.append(c); fragment_keys.extend([key]*len(p))
        return (np.concatenate(points) if points else np.empty((0,3)),
                np.concatenate(colors) if colors else np.empty((0,3)), np.array(fragment_keys))

    def manifest(self):
        return {'options':vars(self.options), 'identity_links_fuse_geometry':False,
                'inactive_hidden_from_live_query':True, 'pending_hidden_from_live_query':True,
                'fragments':[{ 'key':k,'status':f.status,'T_world_submap':f.volume.anchor.tolist(),
                               'blocks':int(f.volume.grid.hashmap().size()),
                               'attribute_capacity_bytes':f.volume.bytes_allocated_attributes(),
                               'observation_mask_bytes':sum(m.nbytes for m in f.masks.values()),
                               'observations':f.observations} for k,f in self.fragments.items()],
                'identities':{k:sorted(v) for k,v in self.identities.items()},'events':self.events}

    def save(self, directory, save_volumes=False):
        directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
        manifest=self.manifest()
        manifest['checkpoint_format']='active-blocks-v1'
        for i,f in enumerate(self.fragments.values()):
            mask_file='observations_%04d.npz'%i
            np.savez_compressed(directory/mask_file,**{'frame_%d'%k:v for k,v in f.masks.items()})
            manifest['fragments'][i]['mask_file']=mask_file
        if save_volumes:
            for i,f in enumerate(self.fragments.values()):
                filename='volume_%04d.npz'%i
                grid=f.volume.grid
                active=grid.hashmap().active_buf_indices().to(o3c.int64)
                # Store active blocks only. Open3D's native loader allocates
                # exactly N slots; GPU collision overflow can lose blocks.
                np.savez_compressed(directory/filename,
                    keys=grid.hashmap().key_tensor()[active].cpu().numpy(),
                    **{k:grid.attribute(k)[active].cpu().numpy() for k in ('tsdf','weight','color')})
                manifest['fragments'][i]['volume_file']=filename
        (directory/'fragments.json').write_text(json.dumps(manifest,indent=2))

    @classmethod
    def load(cls, directory):
        directory=Path(directory)
        manifest=json.loads((directory/'fragments.json').read_text())
        if manifest.get('checkpoint_format')!='active-blocks-v1':
            raise ValueError('unsupported checkpoint format')
        store=cls(TSDFOptions(**manifest['options']))
        for item in manifest['fragments']:
            if 'volume_file' not in item: raise ValueError('manifest-only checkpoint cannot restore geometry')
            store.create(item['key'],item['T_world_submap'])
            f=store.fragments[item['key']]
            with np.load(directory/item['volume_file']) as data:
                n=len(data['keys']); grid=f.volume.grid
                grid.hashmap().reserve(max(store.options.initial_blocks,4*n+1))
                if n:
                    keys=o3c.Tensor(data['keys'],device=f.volume.device)
                    idx,inserted=grid.hashmap().activate(keys)
                    if not inserted.cpu().numpy().all() or int(grid.hashmap().size())!=n:
                        raise RuntimeError('checkpoint restoration lost TSDF blocks')
                    idx=idx.to(o3c.int64)
                    for k in ('tsdf','weight','color'):
                        grid.attribute(k)[idx]=o3c.Tensor(data[k],device=f.volume.device)
            f.status=item['status']; f.observations=item['observations']
            f.seen={x['frame'] for x in f.observations}
            with np.load(directory/item['mask_file']) as data:
                f.masks={int(k[6:]):data[k].copy() for k in data.files}
            f.volume.integrations=len(f.observations)
        store.identities={k:set(v) for k,v in manifest['identities'].items()}
        store.events=manifest['events']
        return store
