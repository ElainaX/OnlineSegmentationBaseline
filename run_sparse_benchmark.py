"""Controlled dense-vs-sparse benchmark on the SAME disjoint submap pipeline."""
import argparse, hashlib, json, random, time, resource, threading, subprocess, os
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
from submap_disjoint.sparse_owner import SparseDisjointSubmapManager

def main():
    p=argparse.ArgumentParser()
    p.add_argument('-c','--config',default='config/scannet_stride10_controlled.yaml')
    p.add_argument('-d','--dataset',required=True)
    p.add_argument('-i','--instance_dir',required=True)
    p.add_argument('-o','--output_dir',required=True)
    p.add_argument('--seq_name',required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--backend',choices=['dense','sparse'],required=True)
    p.add_argument('--verify_sparse',action='store_true')
    p.add_argument('--max_frames',type=int)
    args=p.parse_args()
    out=Path(args.output_dir)/args.seq_name; out.mkdir(parents=True,exist_ok=True)
    if (out/'benchmark.json').exists(): raise RuntimeError('refusing to overwrite finished benchmark')
    wall_start=time.perf_counter()
    gpu_samples=[]; stop=threading.Event()
    def sample_gpu():
        while not stop.is_set():
            try:
                text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
                    '--format=csv,noheader,nounits'],text=True,timeout=3)
                for line in text.splitlines():
                    pid,mb=line.split(',')
                    if int(pid)==os.getpid(): gpu_samples.append(float(mb))
            except Exception: pass
            stop.wait(2)
    sampler=threading.Thread(target=sample_gpu,daemon=True); sampler.start()
    random.seed(27); np.random.seed(27); torch.manual_seed(27); torch.cuda.manual_seed_all(27)
    cfg=load_config(args.config)
    cfg['seg']['depth_shrink']['enabled']=False
    cfg['save']['ckpt_interval']=0; cfg['save']['save_interval']=0
    cfg['submaps']={'enabled':True,'backend':'disjoint_equivalent','baseline_fallback':False}
    cfg['sparse_candidate']={'enabled':args.backend=='sparse','verify':args.verify_sparse}
    dataset=get_dataset(args.dataset,args.instance_dir,cfg,args.device)
    geometry=GeometryScene(cfg,args,dataset,args.device)
    owner_cls=SparseDisjointSubmapManager if args.backend=='sparse' else DisjointSubmapManager
    owner=owner_cls(cfg,args,dataset,geometry,args.device)
    assert not hasattr(geometry,'maskGraph') and not hasattr(geometry,'instance_dict')
    start=time.perf_counter(); last_pose=torch.eye(4); final_id=-1
    frame_seconds=[]; merge_trace=[]
    original_update=owner.update_masks
    def update(frame_id):
        original_update(frame_id)
        merge_trace.append({'frame':int(frame_id),'count':owner.c_mask_num,
            'members':[sorted([list(k) for k in inst.ori_mask_list])
                       for _,inst in sorted(owner.maskGraph.instance_dict.items())]})
    owner.update_masks=update
    for frame_id,(rgb,depth,pose,seg,features,flag) in tqdm(enumerate(DataLoader(dataset,batch_size=1,num_workers=0)),total=len(dataset)):
        if args.max_frames is not None and frame_id>=args.max_frames: break
        frame_start=time.perf_counter()
        final_id=frame_id
        if not torch.isfinite(pose[0]).all(): pose[0]=last_pose
        else: last_pose=pose[0]
        if flag[0]: seg=seg[0]; features=features[0]
        if frame_id % cfg['mapping']['keyframe_freq']==0:
            with torch.no_grad():
                blocks,_=geometry.integrate_frame(frame_id,rgb[0],depth[0],torch.inverse(pose[0]))
                if flag[0] and frame_id % cfg['seg']['seg_add_interval']==0:
                    owner.observe(frame_id,rgb[0],depth[0],pose[0],blocks,seg,features)
        torch.cuda.synchronize()
        frame_seconds.append(time.perf_counter()-frame_start)
    torch.cuda.synchronize(); loop_seconds=time.perf_counter()-start
    owner.save_outputs(final_id,str(out))
    torch.cuda.synchronize()
    stop.set(); sampler.join(timeout=5)
    files=['submap_disjoint/'+f for f in ['owner.py','evidence_graph.py','visibility.py','sparse_owner.py','sparse_edges.py']]
    files+=['run_sparse_benchmark.py']
    report=dict(status='complete',backend=args.backend,verify=args.verify_sparse,seed=27,
        reference='exact_submap_scene0050 (AP 0.185)', frame_count=final_id+1,
        main_loop_seconds=loop_seconds,compute_frame_mean_seconds=float(np.mean(frame_seconds)),
        compute_frame_p50_seconds=float(np.median(frame_seconds)),
        compute_frame_p95_seconds=float(np.percentile(frame_seconds,95)),
        total_seconds=time.perf_counter()-wall_start,
        peak_gpu_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_gpu_torch_reserved_bytes=torch.cuda.max_memory_reserved(),
        peak_gpu_process_sampled_MiB=max(gpu_samples) if gpu_samples else None,
        peak_cpu_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        output_disk_bytes=sum(f.stat().st_size for f in out.rglob('*') if f.is_file()),
        config=cfg,argv=sys.argv,sparse_metrics=getattr(owner,'sparse_audit',None),
        source_hashes={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        git_status=subprocess.check_output(['git','status','--porcelain'],text=True))
    (out/'benchmark.json').write_text(json.dumps(report,indent=2))
    (out/'merge_trace.json').write_text(json.dumps(merge_trace))
    print('BENCHMARK COMPLETE',json.dumps({k:report[k] for k in ['backend','main_loop_seconds','total_seconds','peak_cpu_rss_bytes','peak_gpu_process_sampled_MiB']}),flush=True)

if __name__=='__main__': main()

