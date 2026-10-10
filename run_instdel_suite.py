import json, os, subprocess, sys, time, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

OUT=Path('/root/autodl-tmp/output/igsubmap_instdel_suite_20261010')
REPO=Path('/root/OnlineSegmentationBaseline_ig_deletion')
CFGROOT=Path('/root/autodl-tmp/output/six_scene_extension_20261008/configs')
R3=Path('/root/autodl-tmp/3RScan/3RScan_Dynamic_Additional_20261008')
SN=Path('/root/autodl-tmp/ScanNet_additional_0549_similar')
SEGS=Path('/root/autodl-tmp/output/six_scene_extension_20261008/seg_result')
PY=sys.executable
SCENES=[]
for seq in ['pair_05_095821f7_095821f9','pair_06_0988ea72_9766cbe5','pair_07_0cac7540_0cac7532','pair_08_0cac75dc_0cac75de']:
    SCENES.append(dict(seq=seq,dataset=str(R3/seq/'frames'),masks=str(SEGS/seq),config=str(CFGROOT/(seq+'.yaml')),dataset_kind='3RScan'))
for seq in ['scene0089_00','scene0243_00']:
    SCENES.append(dict(seq=seq,dataset=str(SN/'prepared'/seq/'frames'),masks=str(SN/'seg_result'/seq),config=str(CFGROOT/(seq+'.yaml')),dataset_kind='ScanNet'))

def run(spec,gpu):
    seq=spec['seq']; dest=OUT/'delete'; dest.mkdir(parents=True,exist_ok=True)
    out=dest/seq
    cmd=[PY,'-u','run_instdel.py','--config',spec['config'],'--dataset',spec['dataset'],'--instance_dir',spec['masks'],
         '--output_dir',str(dest),'--seq_name',seq,'--merge_selection','disjoint_pairs','--correction','delete']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTHONPATH=str(REPO))
    started=time.time(); log=OUT/(seq+'.run.log')
    row=dict(seq=seq,dataset=spec['dataset_kind'],gpu=gpu,command=cmd,status='running',started=started)
    (OUT/'status').mkdir(parents=True,exist_ok=True); (OUT/'status'/f'{seq}.json').write_text(json.dumps(row,indent=2))
    with log.open('w') as f: rc=subprocess.call(cmd,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
    row.update(returncode=rc,seconds=time.time()-started,status='complete' if rc==0 else 'failed',log=str(log))
    if rc==0 and (out/'run_metadata.json').exists(): row['run_metadata']=json.loads((out/'run_metadata.json').read_text())
    if rc==0 and (out/'correction'/'summary.json').exists(): row['correction_summary']=json.loads((out/'correction'/'summary.json').read_text())
    (OUT/'status'/f'{seq}.json').write_text(json.dumps(row,indent=2))
    print(json.dumps({'seq':seq,'gpu':gpu,'status':row['status'],'seconds':row['seconds']}),flush=True)
    return row

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'scenes.json').write_text(json.dumps(SCENES,indent=2))
    if (OUT/'done.json').exists(): raise SystemExit('refusing to overwrite completed suite')
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run,spec,[4,7][i%2]) for i,spec in enumerate(SCENES)]
        rows=[f.result() for f in futures]
    (OUT/'done.json').write_text(json.dumps(rows,indent=2))
    if any(r['status']!='complete' for r in rows): raise SystemExit(1)
