"""Read frozen TSDFs, verify original export, then retain >=1 observation.

No association rerun, new matching, GT geometry, or in-place output replacement.
Semantic embeddings are omitted: these outputs are class-agnostic AP only.
"""
import argparse, copy, json, time, subprocess
from pathlib import Path
from types import SimpleNamespace
import tempfile
import numpy as np
import torch
import open3d as o3d
from submap_disjoint.it_adapter import InstanceTSDFAdapter
from submap_disjoint.independent_tsdf import IndependentTSDFStore,TSDFOptions

p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
a=p.parse_args();source=Path(a.source);output=Path(a.output)
assert not output.exists(), 'Do not overwrite exported results'
manifest=json.loads((source/'it_manifest.json').read_text())
metadata=json.loads((source/'run_metadata.json').read_text());cfg=metadata['config']
tracks=json.loads((source/'ig_audit/submap_migration.json').read_text())['instances']
assert metadata['merge_audit']['merge_calls']>=5
threshold=cfg['seg']['mask_weight_threshold']
assert threshold==5, 'Re-export requires the fully established five-observation gate'
ids=[int(x['id']) for x in tracks];n=max(ids)+1
weights=torch.zeros(n)
instances={}
for x in tracks:
    weights[x['id']]=len(x['observations'])
    instances[x['id']]=SimpleNamespace(get_semantic_feature=torch.zeros(cfg['mask']['feature_dim']))
max_gid=max(g for f in manifest['fragments'] for g in f['global_ids'])
mapping=[-1]*(max_gid+1);fragments={f['key']:f for f in manifest['fragments']}
for mid,keys in manifest['identities'].items():
    for key in keys:
        for g in fragments[key]['global_ids']:mapping[g]=int(mid)

def save_pc(xyz,rgb,path):
    pc=o3d.geometry.PointCloud();pc.points=o3d.utility.Vector3dVector(xyz)
    pc.colors=o3d.utility.Vector3dVector(rgb);o3d.io.write_point_cloud(path,pc)

owner=SimpleNamespace(device='cuda:0',global2merged_mask_id=mapping,c_mask_num=n,
    merged_mask_weight=weights,merge_time=5,maskGraph=SimpleNamespace(instance_dict=instances),save_pc=save_pc)

def adapter(directory,threshold):
    x=InstanceTSDFAdapter.__new__(InstanceTSDFAdapter)
    x.owner=owner;x.cfg=cfg;x.directory=directory;x.packets=[]
    options=dict(manifest['options']);options['extraction_weight']=threshold
    x.options=TSDFOptions(**options);x.fragments=manifest['fragments'];x.windows=manifest['windows']
    x.audit=copy.deepcopy(manifest['audit'])
    return x

# Reconstruct the exporter state from recorded predicted identities. First
# reproduce the ORIGINAL output bit-for-bit, before changing any threshold.
with tempfile.TemporaryDirectory(prefix='it_export_parity_') as temp:
    adapter(source/'it_fragments',1.).export(metadata['last_processed_frame'],temp)
    with np.load(Path(temp)/'ckpt_final.npz') as recovered, np.load(source/'ckpt_final.npz') as original:
        assert np.array_equal(recovered['pred_masks'],original['pred_masks']), 'original masks failed export parity'
    x=np.asarray(o3d.io.read_point_cloud(str(Path(temp)/'final.ply')).points)
    y=np.asarray(o3d.io.read_point_cloud(str(source/'final.ply')).points)
    assert np.array_equal(x,y), 'original surface failed export parity'
print('ORIGINAL_EXPORT_PARITY_PASS',source.name,flush=True)

started=time.perf_counter();output.mkdir(parents=True)
new_fragments=output/'it_fragments';new_fragments.mkdir()
surface_counts=[]
for i,f in enumerate(manifest['fragments']):
    old=source/'it_fragments'/f['key'];target=new_fragments/f['key'];target.mkdir()
    store=IndependentTSDFStore.load(old)
    volume=store.fragments[f['key']].volume
    xyz,rgb=volume.surface(.5)
    np.savez_compressed(target/'surface.npz',points=xyz.astype(np.float32),colors=rgb.astype(np.float32))
    surface_counts.append((len(xyz),f['surface_points']))
    local=json.loads((old/'fragments.json').read_text());local['options']['extraction_weight']=.5
    for item in local['fragments']:
        for key in ['volume_file','mask_file']:(target/item[key]).symlink_to(old/item[key])
    (target/'fragments.json').write_text(json.dumps(local,indent=2))
    f['surface_points']=len(xyz)
    del volume,store
    if i%200==0:print('REEXTRACT',source.name,i,len(manifest['fragments']),flush=True)

fixed=adapter(new_fragments,.5)
report=fixed.export(metadata['last_processed_frame'],output)
report['surface_threshold_semantics']='Open3D weight > 0.5 retains voxels with at least one integer-weight observation'
report['extraction_weight_note']='Strict threshold 0.5; initial strict threshold 1 excluded single-observation voxels'
report['reexport_source']=str(source)
report['original_export_parity']='point coordinates and instance masks exactly reproduced before threshold change'
report['semantic_embeddings_omitted']=True
report['audit']['raw_surface_points']=sum(n for n,_ in surface_counts)
report['reexport_seconds']=time.perf_counter()-started
report['new_surface_fragments_from_empty']=sum(n>0 and old==0 for n,old in surface_counts)
report['remaining_empty_fragments']=sum(n==0 for n,_ in surface_counts)
report['code_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
(output/'it_manifest.json').write_text(json.dumps(report,indent=2))
# Optional in the original evaluator. Do not pretend to restore unsaved
# semantic vectors when only class-agnostic masks are needed for this audit.
with np.load(output/'ckpt_final.npz') as data:
    arrays={k:data[k].copy() for k in data.files if k!='pred_sem_features'}
np.savez(output/'ckpt_final.npz',**arrays)
metadata.update(export_only_repair=True,reexport_seconds=report['reexport_seconds'],
    source_main_loop_seconds=metadata['main_loop_seconds'],surface_strict_weight_threshold=.5,
    runtime_note='Main-loop time is the original threshold=1 run; 0.5 re-export is timed separately',
    export_points=report['audit']['export_points'])
(output/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
print('REEXPORT_COMPLETE',json.dumps({k:report[k] for k in ['reexport_seconds','new_surface_fragments_from_empty','remaining_empty_fragments']}),flush=True)
