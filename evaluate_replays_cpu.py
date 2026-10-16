"""Recover evaluation only; do not repeat or alter any local reconstruction."""
import os,sys,time,json,pathlib
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
import torch,open3d as o3d
torch.set_num_threads(4);torch.Tensor.cuda=lambda self,*args,**kwargs:self
root=pathlib.Path('/root/autodl-tmp/multires_selective2_20261013')
sys.path.insert(0,'/root/OnlineAnySegBaseline_original')
sys.argv=['eval','--result_dir',str(root),'--gt_dir',str(root),'--output_file',str(root/'diagnostic_eval.txt')]
from eval import evaluate_seqs as ev
refs=json.loads(pathlib.Path('/root/autodl-tmp/multires_extra2_20261013/manifest.json').read_text())
summary=[]
for ref in refs:
    seq=ref['seq'];out=root/seq;start=time.monotonic();report=json.loads((out/'report.json').read_text());results={}
    for name in ['local25','local20','selected']:
        p=out/name/seq;gtfile=pathlib.Path(ref['gtseg'])/(seq+'.txt');gtpc=pathlib.Path(ref['gtroot'])/seq/(seq+'_vh_clean_2.ply')
        gt,pred=ev.assign_instances_for_scan(seq,str(p/'ckpt_final.npz'),str(gtfile),o3d.io.read_point_cloud(str(p/'final.ply')),o3d.io.read_point_cloud(str(gtpc)))
        avg=ev.compute_averages(ev.evaluate_matches({seq:{'gt':gt,'pred':pred}}))
        results[name]=dict(AP=avg['all_ap'],AP50=avg['all_ap_50%'],AP25=avg['all_ap_25%'])
        print('RESULT',seq,name,results[name],flush=True)
    item=dict(scene=seq,status='complete',results=results,evaluation_seconds=time.monotonic()-start,
              construction_seconds=report['construction_seconds'],selection_seconds=report['selection_seconds'],
              choices={k:sum(r['choice']==k for r in report['records']) for k in ['ig','local25','local20']},
              gpu_sampling=report['gpu_sampling'],
              recovery_note='Initial runner expected wrong metric filename. Reconstructed outputs reused unchanged; official evaluator run on CPU.')
    summary.append(item);(root/'evaluation_recovery.json').write_text(json.dumps(summary,indent=2))
print('COMPLETE',json.dumps(summary),flush=True)
