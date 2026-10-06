"""Controlled geometry deletion experiment, not an instance-AP benchmark.

All methods receive the same RGBD, camera poses, oracle assignment and change
time. Replay is a deliberately strong global TSDF baseline, not a straw man.
"""
import argparse, json, time, resource, gc
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d
from submap_disjoint.independent_tsdf import TSDFOptions, LocalTSDF, IndependentTSDFStore


def sync(device):
    if device.upper().startswith('CUDA'): o3d.core.cuda.synchronize()


def backproject(depth, K, pose, mask=None):
    v,u=np.indices(depth.shape)
    valid=depth>0
    if mask is not None: valid &= mask
    d=depth[valid]
    xyz=np.column_stack(((u[valid]-K[0,2])*d/K[0,0],(v[valid]-K[1,2])*d/K[1,1],d))
    return xyz@pose[:3,:3].T+pose[:3,3]


def compact(points, voxel=.0125):
    if not len(points): return np.empty((0,3))
    _,i=np.unique(np.floor(points/voxel).astype(np.int32),axis=0,return_index=True)
    return points[i]


def distance(query, reference):
    if not len(reference): return np.full(len(query),np.inf)
    return cKDTree(reference).query(query,workers=4)[0]


def render_synthetic(wall_z, box_z, index, has_object=True):
    h,w=96,128; K=np.array([[100.,0,63.5],[0,100.,47.5],[0,0,1]])
    pose=np.eye(4); pose[0,3]=.75*np.sin(index*.43)
    pose[1,3]=.1*np.cos(index*.31)
    v,u=np.indices((h,w)); rays=np.stack(((u-K[0,2])/100,(v-K[1,2])/100,np.ones((h,w))),-1)
    depth=np.full((h,w),wall_z,dtype=np.float32)
    target=np.zeros((h,w),bool)
    if has_object:
        lower=np.array([-.25,-.3,box_z]); upper=np.array([.25,.3,box_z+.15])
        # Ray/AABB slab intersection. Pixel rays are parameterized by camera Z.
        with np.errstate(divide='ignore',invalid='ignore'):
            ta=(lower-pose[:3,3])/rays; tb=(upper-pose[:3,3])/rays
        enter=np.minimum(ta,tb).max(-1); leave=np.maximum(ta,tb).min(-1)
        target=(leave>=enter)&(enter>0)&(enter<depth)
        depth[target]=enter[target]
    rgb=np.full((h,w,3),150,np.uint8); rgb[target]=[230,40,40]
    return dict(frame=index,rgb=rgb,depth=depth,K=K,pose=pose,mask=target,
                source={'kind':'synthetic_exact_ray_aabb','has_object':has_object})


def run_case(before, after, out, options, extra_metadata=None):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists(): raise RuntimeError('refusing to overwrite completed experiment')
    target=compact(np.concatenate([backproject(f['depth'],f['K'],f['pose'],f['mask']) for f in before]))
    background=compact(np.concatenate([backproject(f['depth'],f['K'],f['pose'],~f['mask']) for f in before]))
    current=compact(np.concatenate([backproject(f['depth'],f['K'],f['pose']) for f in after]))
    tolerance=1.5*options.voxel_m
    # Stale probes are old observed surfaces demonstrably absent from current
    # observed geometry. Real-data runner further restricts these using free rays.
    stale=target[distance(target,current)>2*tolerance]
    if extra_metadata and 'free_space_stale_probes' in extra_metadata:
        stale=extra_metadata.pop('free_space_stale_probes')
    # Background from historical observations; real data can exclude other
    # changed instances via the optional static probe set.
    static=background
    if extra_metadata and 'static_probes' in extra_metadata:
        static=extra_metadata.pop('static_probes')
    if not len(target): raise ValueError('no observed oracle target')
    bounds=(target.min(0),target.max(0))
    np.savez_compressed(out/'evaluation_probes.npz',stale=stale,static=static,current=current,target=target)
    results=[]
    for method in ('global_no_repair','global_clear_tight','global_clear_band','global_clear_target_support','global_replay','independent_deactivate'):
        gc.collect(); sync(options.device); start=time.perf_counter()
        independent=method=='independent_deactivate'
        model=IndependentTSDFStore(options) if independent else LocalTSDF(options)
        if independent:
            model.create('background',identity='static_background')
            model.create('old_object_state',identity='object_identity_unknown')
        for f in before:
            if independent:
                for key,mask in [('background',~f['mask']),('old_object_state',f['mask'])]:
                    model.integrate(key,f['frame'],f['rgb'],f['depth'],f['K'],f['pose'],mask,f['source'])
            else: model.integrate(f['rgb'],f['depth'],f['K'],f['pose'])
        sync(options.device); build_s=time.perf_counter()-start
        def points(): return model.surface()[0]
        def allocated():
            return (sum(f.volume.bytes_allocated_attributes() for f in model.fragments.values())
                    if independent else model.bytes_allocated_attributes())
        stage=[]; baseline_static=None
        def measure(name, n_new):
            nonlocal baseline_static
            sync(options.device); ts=time.perf_counter(); p=points(); sync(options.device)
            extraction_s=time.perf_counter()-ts
            static_hits=distance(static,p)<=tolerance
            if baseline_static is None: baseline_static=static_hits.copy()
            stale_hits=distance(stale,p)<=tolerance
            row={'stage':name,'new_frames':n_new,'surface_points':len(p),
                 'stale_residual_fraction':float(stale_hits.mean()) if len(stale) else None,
                 'historical_static_coverage':float(static_hits.mean()) if len(static) else None,
                 'lost_previously_supported_static_fraction':float((baseline_static & ~static_hits).sum()/max(1,baseline_static.sum())),
                 'current_observed_surface_coverage':float((distance(current,p)<=tolerance).mean()),
                 'attribute_capacity_bytes':allocated(),'extraction_seconds':extraction_s}
            stage.append(row)
            pc=o3d.geometry.PointCloud(); pc.points=o3d.utility.Vector3dVector(p)
            if len(p): o3d.io.write_point_cloud(str(out/(method+'_'+name+'.ply')),pc)
        measure('before',0)
        if extra_metadata and extra_metadata.get('data')=='synthetic' and stage[0]['historical_static_coverage'] < .8:
            raise RuntimeError('invalid synthetic control: background reconstruction failed before deletion')
        support=None; support_prepare_s=0.
        if method=='global_clear_target_support':
            sync(options.device); ts=time.perf_counter()
            support=LocalTSDF(options)
            for f in before: support.integrate(f['rgb'],f['depth'],f['K'],f['pose'],f['mask'])
            sync(options.device); support_prepare_s=time.perf_counter()-ts
        sync(options.device); repair_start=time.perf_counter(); cleared=0
        if method=='global_clear_target_support':
            cleared=model.clear_observed_support(support)
        elif method.startswith('global_clear'):
            pad=options.voxel_m*options.trunc_voxels if method.endswith('band') else 0.
            cleared=model.clear_aabb(bounds[0]-pad,bounds[1]+pad)
        elif method=='global_replay':
            del model; gc.collect()
            model=LocalTSDF(options)
            for f in before:
                model.integrate(f['rgb'],f['depth'],f['K'],f['pose'],~f['mask'])
        elif independent:
            model.set_status('old_object_state','frozen','end of first temporal state')
            model.set_status('old_object_state','inactive','oracle known disappeared/moved target')
            model.create('current_epoch',identity='new_states_not_reidentified')
        sync(options.device); repair_s=time.perf_counter()-repair_start
        measure('immediate',0)
        support_bytes=support.bytes_allocated_attributes() if support is not None else 0
        del support
        after_start=time.perf_counter(); integration_s=0.
        for n,f in enumerate(after,1):
            sync(options.device); tick=time.perf_counter()
            if independent:
                model.integrate('current_epoch',f['frame'],f['rgb'],f['depth'],f['K'],f['pose'],
                                f['depth']>0,f['source'])
            else: model.integrate(f['rgb'],f['depth'],f['K'],f['pose'])
            sync(options.device); integration_s+=time.perf_counter()-tick
            if n in (1,5,10,len(after)): measure('after_%d'%n,n)
        if independent: model.save(out/'independent_fragments',save_volumes=False)
        results.append({'method':method,'build_seconds':build_s,'repair_seconds':repair_s,
                        'oracle_support_prepare_seconds':support_prepare_s,'oracle_support_attribute_bytes':support_bytes,
                        'after_integration_seconds':integration_s,'cleared_voxels':cleared,'stages':stage})
        (out/'partial_results.json').write_text(json.dumps(results,indent=2))
        print(method,'repair_s',round(repair_s,5),'immediate',stage[1],flush=True)
        del model; gc.collect()
    report={'protocol':'oracle target, equal input, geometry deletion; NOT AP / NOT automatic detection',
            'options':vars(options),'before_frames':len(before),'after_frames':len(after),
            'tolerance_m':tolerance,'stale_probe_count':len(stale),'static_probe_count':len(static),
            'current_probe_count':len(current),'target_aabb':[b.tolist() for b in bounds],
            'history_rgbd_payload_bytes':sum(f['rgb'].nbytes+f['depth'].nbytes+f['mask'].nbytes for f in before),
            'history_note':'shared in-memory input cache for experiment; replay requires source history in deployment',
            'memory_note':'per-method grid attribute capacity, excluding allocator/hash overhead; process peak is shared across methods',
            'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'metadata':extra_metadata,'results':results}
    (out/'results.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True); p.add_argument('--device',default='CUDA:0')
    a=p.parse_args()
    # Avoid the marching-cubes exact-zero degeneracy of perfectly grid-aligned
    # infinite planes; real depth has noise. Validate reconstruction before repair.
    for name,wall,box in [('isolated',2.417,1.513),('near_wall',2.203,2.013)]:
        before=[render_synthetic(wall,box,i) for i in range(30)]
        after=[render_synthetic(wall,box,30+i,False) for i in range(12)]
        run_case(before,after,Path(a.out)/name,TSDFOptions(device=a.device),
                 {'data':'synthetic','exact_target_assignment':True,'wall_z':wall,'box_z':box})
