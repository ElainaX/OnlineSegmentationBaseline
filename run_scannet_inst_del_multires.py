"""Four-scene ScanNet InstDel resolution comparison, max 1000 sampled frames."""
import json, math, os, re, shutil, subprocess, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import yaml
OLD=Path('/root/autodl-tmp/output/six_scene_extension_20261008')
MANIFESTS=[OLD/'scannet_valid_manifest.json',OLD/'scannet_replacement_manifest.json']
ROOT=Path('/root/autodl-tmp/inst_del_multires_scan_20261012_retry')
REPO=Path('/root/OnlineSegmentationBaseline_ig_inst_del_multires_20261012')
EVAL=Path('/root/OnlineAnySegBaseline_original'); PY='/root/miniconda3/envs/OASeg/bin/python'; GPUS=[4,7]
TARGETS={'scene0089_00','scene0243_00','scene0568_00','scene0608_00'}
# scene0243 has a pathological dense frame around frame_220 at stride 10;
# use a wider, still mask-aligned interval so one scene cannot monopolize a GPU.
STRIDE_MULTIPLIER={'scene0243_00': 3}
VARIANTS={
 'baseline_25mm': ['--global-voxel-m','0.025'],
 'adaptive_15mm': ['--global-voxel-m','0.025','--fine-voxel-m','0.015','--multires','--online-fine-evidence'],
 'baseline_40mm': ['--global-voxel-m','0.04'],
 'adaptive_25mm': ['--global-voxel-m','0.04','--fine-voxel-m','0.025','--multires','--online-fine-evidence'],
}
refs=[]
for p in MANIFESTS: refs.extend(json.loads(p.read_text())['rows'])
refs=[r for r in refs if r['seq'] in TARGETS]
if {r['seq'] for r in refs}!=TARGETS: raise RuntimeError('missing ScanNet target')
ROOT.mkdir(parents=True,exist_ok=True)
if (ROOT/'results.json').exists(): raise RuntimeError('results exists')
def numeric_files(d):
 out={}
 for p in Path(d).iterdir():
  if p.is_file() and p.stem.isdigit(): out[int(p.stem)]=p
 return out
def prepare(ref):
 seq=ref['seq']; cfg=yaml.safe_load(Path(ref['config']).read_text());
 raw=len(list((Path(ref['frames'])/'color').iterdir()))
 masks=numeric_files(Path(ref['masks'])/'mask'); embeds=numeric_files(Path(ref['masks'])/'mask_embeddings')
 ids=sorted(set(masks)&set(embeds));
 if not ids: raise RuntimeError('no numeric mask/embed files for '+seq)
 key=max(1,int(cfg.get('mapping',{}).get('keyframe_freq',1))); seg=max(1,int(cfg.get('seg',{}).get('seg_add_interval',1)))
 quantum=math.lcm(key,seg); raw_stride=max(1,math.ceil(raw/1000)); stride=max(quantum,math.ceil(raw_stride/quantum)*quantum)
 stride=max(stride, stride*STRIDE_MULTIPLIER.get(seq,1))
 start=ids[0]; selected=[i for i in range(start,raw,stride) if i in masks and i in embeds]
 if len(selected)>1000: raise AssertionError((seq,len(selected),stride))
 out=ROOT/'sampled_masks'/seq; shutil.rmtree(out,ignore_errors=True); (out/'mask').mkdir(parents=True); (out/'mask_embeddings').mkdir()
 for i in selected:
  shutil.copy2(masks[i],out/'mask'/masks[i].name); shutil.copy2(embeds[i],out/'mask_embeddings'/embeds[i].name)
 (out/'sample_manifest.json').write_text(json.dumps({'seq':seq,'raw_frame_count':raw,'selected_frame_ids':selected,'effective_frame_count':len(selected),'stride':stride,'mask_interval':seg,'keyframe_freq':key,'mask_files':[masks[i].name for i in selected]},indent=2))
 return {'frames':raw,'selected':selected,'stride':stride,'mask_dir':str(out),'manifest':str(out/'sample_manifest.json')}
def run(cmd,log,env,cwd):
 s=time.time()
 with open(log,'w') as f: rc=subprocess.call(cmd,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
 if rc: raise RuntimeError(Path(log).read_text(errors='replace')[-5000:])
 return time.time()-s
def evaluate(seq,root,ref,env,log):
 cmd=[PY,'-m','eval.evaluate_seqs','--result_dir',str(root),'--seq_name',seq,'--gt_dir',ref['gtroot'],'--gt_pc_pattern','%s/%s_vh_clean_2.ply','--gt_seg_dir',ref['gtseg'],'--gt_seg_pattern','%s.txt','--output_file',str(root/(seq+'_class_agnostic.txt'))]
 run(cmd,log,dict(env,PYTHONPATH=str(EVAL)),EVAL)
 m=re.search(r'average\s*:\s*([\d.nan]+)\s+([\d.nan]+)\s+([\d.nan]+)',Path(log).read_text(errors='replace'))
 if not m: raise RuntimeError('AP line missing: '+str(log))
 return dict(zip(['AP','AP50','AP25'],map(float,m.groups())))
records=[]; lock=threading.Lock()
def save():
 t=ROOT/'results.tmp'; t.write_text(json.dumps(records,indent=2)); t.replace(ROOT/'results.json')
def one(ref,gpu):
 seq=ref['seq']; sample=prepare(ref); env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTHONPATH=str(REPO))
 row={'seq':seq,'gpu':gpu,'status':'running','sampling':sample,'variants':{}}
 with lock: records.append(row); save()
 try:
  for name,extra in VARIANTS.items():
   base=ROOT/name; base.mkdir(exist_ok=True)
   cmd=[PY,'run_instdel_multires.py','--config',ref['config'],'--dataset',ref['frames'],'--instance_dir',sample['mask_dir'],'--output_dir',str(base),'--seq_name',seq,'--merge_selection','disjoint_pairs','--correction','delete','--frame-ids-file',sample['manifest']]
   # The runner expects a JSON list for frame-ids-file; create a companion list.
   frame_file=ROOT/'frame_ids'/f'{seq}.json'; frame_file.parent.mkdir(exist_ok=True); frame_file.write_text(json.dumps(sample['selected']))
   cmd[-1]=str(frame_file); cmd += extra
   log=ROOT/(seq+'_'+name+'.run.log'); row['variants'][name]={'command':cmd,'sampling':sample}; row['variants'][name]['run_seconds']=run(cmd,log,env,REPO)
   out=base/seq; meta=json.loads((out/'run_metadata.json').read_text()); row['variants'][name]['run_metadata']=meta; row['variants'][name]['correction_summary']=meta.get('correction_summary'); row['variants'][name]['online_fine_evidence']=meta.get('online_fine_evidence_audit'); row['variants'][name]['AP']=evaluate(seq,base,ref,env,ROOT/(seq+'_'+name+'.eval.log'))
   with lock: save()
  row['status']='complete'
 except Exception as e: row['status']='failed'; row['error']=repr(e)
 finally:
  with lock: save()
def gpu_worker(gpu, assigned):
 for ref in assigned:
  one(ref, gpu)

# Keep at most one scene per GPU at a time, but cover all four scenes.
with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
 futures=[pool.submit(gpu_worker, gpu, refs[i::len(GPUS)]) for i,gpu in enumerate(GPUS)]
 for f in futures: f.result()
(ROOT/'done.json').write_text(json.dumps({'complete':sum(x['status']=='complete' for x in records),'failed':sum(x['status']=='failed' for x in records),'total':len(records),'finished':time.time()},indent=2))
