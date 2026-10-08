"""A prereported extra IG-off repeat: zero-erasure AP changed in pair01."""
import json,os,subprocess,time,re
from pathlib import Path
R=Path('/root/autodl-tmp/output/ig_correction_20261008')
repo=Path('/root/OnlineSegmentationBaseline_ig_correction');original=Path('/root/OnlineAnySegBaseline_original')
py='/root/miniconda3/envs/OASeg/bin/python'
ref=next(r for r in json.loads((R/'sources.json').read_text()) if r['seq'].startswith('pair_01'))
dest=R/'off_repeat';dest.mkdir(exist_ok=False);seq=ref['seq']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='6',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
cmd=[py,'run_ig_correction.py','--config',ref['config'],'--dataset',ref['frames'],'--instance_dir',ref['masks'],
     '--output_dir',str(dest),'--seq_name',seq,'--merge_selection','disjoint_pairs','--correction','off']
row=dict(seq=seq,variant='off_repeat',reason='pair01 AP differed although correction executed zero erasures',command=cmd,status='mapping')
(dest/'result.json').write_text(json.dumps(row,indent=2));started=time.time()
with (dest/'run.log').open('w') as f:rc=subprocess.call(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT)
row.update(returncode=rc,wall_seconds=time.time()-started)
if not rc:
    row['metadata']=json.loads((dest/seq/'run_metadata.json').read_text())
    cmd=[py,'-m','eval.evaluate_seqs','--result_dir',str(dest),'--seq_name',seq,'--gt_dir',ref['gtroot'],
         '--gt_pc_pattern','%s/%s_vh_clean_2.ply','--gt_seg_dir',ref['gtseg'],'--gt_seg_pattern','%s.txt','--output_file',str(dest/'class_agnostic.txt')]
    with (dest/'eval.log').open('w') as f:rc=subprocess.call(cmd,cwd=original,env=dict(env,PYTHONPATH=str(original)),stdout=f,stderr=subprocess.STDOUT)
    m=re.search(r'average\s*:\s*([\d.nan]+)\s+([\d.nan]+)\s+([\d.nan]+)',(dest/'eval.log').read_text())
    if not rc and m:row.update(dict(zip(['AP','AP50','AP25'],map(float,m.groups()))));row['status']='complete'
    else:row['status']='evaluation_failed'
else:row['status']='mapping_failed'
(dest/'result.json').write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
