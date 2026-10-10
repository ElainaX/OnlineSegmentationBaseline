"""Invoke the untouched original evaluator and save full precision results."""
import json, runpy, sys, hashlib
from pathlib import Path
import numpy as np
import open3d as o3d

ORIGINAL=Path('/root/OnlineAnySegBaseline_original')
DATA=Path('/root/autodl-tmp/3RScan/3RScan_Dynamic_Additional_20261008')
OUT=Path('/root/autodl-tmp/output/gap_v2_20261010')
SEQS=['pair_09_0cac7676_7272e17c','pair_10_10b17940_5630cfe7']
ROOTS={
 'Baseline':Path('/root/autodl-tmp/output/six_scene_extension_20261008/baseline'),
 'IGSubmap':Path('/root/autodl-tmp/output/ig_delete_highchange_20261010/off'),
 'ITSubmap':Path('/root/autodl-tmp/output/six_scene_extension_20261008/IT_disjoint'),
 'IGSubmap_delete':Path('/root/autodl-tmp/output/ig_delete_highchange_20261010/delete'),
 'ITSubmap_delete':Path('/root/autodl-tmp/output/it_delete_highchange_20261010'),
}

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 sys.path.insert(0,str(ORIGINAL))
 sys.argv=['evaluate_seqs.py','--result_dir',str(OUT),'--gt_dir',str(DATA),'--output_file',str(OUT/'standard_class_agnostic.txt')]
 e=runpy.run_path(str(ORIGINAL/'eval/evaluate_seqs.py'),run_name='standard_ap_adapter')
 result={'evaluator':str(ORIGINAL/'eval/evaluate_seqs.py'),'evaluator_sha256':digest(ORIGINAL/'eval/evaluate_seqs.py'),'sequences':{}}
 for seq in SEQS:
  result['sequences'][seq]={}
  for method,base in ROOTS.items():
   root=base/seq; print(seq,method,flush=True)
   gt,pred=e['assign_instances_for_scan'](seq,str(root/'ckpt_final.npz'),str(DATA/'gt_seg'/(seq+'.txt')),
      o3d.io.read_point_cloud(str(root/'final.ply')),o3d.io.read_point_cloud(str(DATA/seq/(seq+'_vh_clean_2.ply'))),save_seg_gt=False)
   scores=e['evaluate_matches']({seq:dict(gt=gt,pred=pred)})
   a=e['compute_averages'](scores)
   row=dict(AP=float(a['all_ap']),AP50=float(a['all_ap_50%']),AP25=float(a['all_ap_25%']),
     root=str(root),thresholds=e['opt'].overlaps.tolist(),per_threshold=[float(x) for x in scores[0,0]],
     GT_valid=sum(g['instance_id']>=1000 and g['vert_count']>=100 for gs in gt.values() for g in gs),
     inputs={str(p):digest(p) for p in [root/'final.ply',root/'ckpt_final.npz',DATA/'gt_seg'/(seq+'.txt'),DATA/seq/(seq+'_vh_clean_2.ply')]})
   result['sequences'][seq][method]=row
   (OUT/'standard_ap.json').write_text(json.dumps(result,indent=2,allow_nan=False))
   print({k:row[k] for k in ['AP','AP50','AP25','GT_valid']},flush=True)
 return result

if __name__=='__main__': main()
