"""Baseline-disjoint equivalent migration into a sole submap instance owner."""
import argparse, hashlib, json, random, time
from pathlib import Path
import sys
sys.path.append('third_party/FCGF')
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from Dataset.dataset import get_dataset
from tool.config import load_config
from submap_disjoint.geometry import GeometryScene
from submap_disjoint.owner import DisjointSubmapManager

def main():
    p=argparse.ArgumentParser()
    p.add_argument('-c','--config',default='config/scannet_stride10_controlled.yaml')
    p.add_argument('-d','--dataset',required=True)
    p.add_argument('-i','--instance_dir',required=True)
    p.add_argument('-o','--output_dir',required=True)
    p.add_argument('--seq_name',default='scene0050_00')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--max_frames',type=int)
    args=p.parse_args()
    random.seed(27); np.random.seed(27); torch.manual_seed(27); torch.cuda.manual_seed_all(27)
    cfg=load_config(args.config)
    cfg['seg']['depth_shrink']['enabled']=False
    cfg['save']['ckpt_interval']=0; cfg['save']['save_interval']=0
    cfg['submaps']={'enabled':True,'backend':'disjoint_equivalent','baseline_fallback':False}
    dataset=get_dataset(args.dataset,args.instance_dir,cfg,args.device)
    out=Path(args.output_dir)/args.seq_name; out.mkdir(parents=True,exist_ok=True)
    geometry=GeometryScene(cfg,args,dataset,args.device)
    owner=DisjointSubmapManager(cfg,args,dataset,geometry,args.device)
    assert not hasattr(geometry,'maskGraph') and not hasattr(geometry,'instance_dict')
    start=time.perf_counter(); last_pose=torch.eye(4); final_id=-1
    for frame_id,(rgb,depth,pose,seg,features,flag) in tqdm(enumerate(DataLoader(dataset,batch_size=1,num_workers=0)),total=len(dataset)):
        if args.max_frames is not None and frame_id>=args.max_frames: break
        final_id=frame_id
        if not torch.isfinite(pose[0]).all(): pose[0]=last_pose
        else: last_pose=pose[0]
        if flag[0]: seg=seg[0]; features=features[0]
        if frame_id % cfg['mapping']['keyframe_freq']==0:
            with torch.no_grad():
                blocks,_=geometry.integrate_frame(frame_id,rgb[0],depth[0],torch.inverse(pose[0]))
                if flag[0] and frame_id % cfg['seg']['seg_add_interval']==0:
                    owner.observe(frame_id,rgb[0],depth[0],pose[0],blocks,seg,features)
    torch.cuda.synchronize(); loop_seconds=time.perf_counter()-start
    owner.save_outputs(final_id,str(out))
    torch.cuda.synchronize()
    files=['submap_disjoint/owner.py','submap_disjoint/evidence_graph.py','submap_disjoint/visibility.py','run_submap_disjoint.py']
    manifest=dict(status='complete',reference='disjoint_pairs_reference_current',seed=27,
                  frame_count=final_id+1,main_loop_seconds=loop_seconds,total_seconds=time.perf_counter()-start,
                  peak_gpu_bytes=torch.cuda.max_memory_allocated(),config=cfg,argv=sys.argv,
                  source_hashes={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files})
    (out/'migration_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('MIGRATION COMPLETE',loop_seconds,manifest['total_seconds'],flush=True)
if __name__=='__main__': main()
