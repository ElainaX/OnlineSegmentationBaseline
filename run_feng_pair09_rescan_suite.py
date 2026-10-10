import json,os,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

OUT=Path('/root/autodl-tmp/output/feng_pair09_rescan_20261010')
SEQ='pair_09_0cac7676_7272e17c'
def run(mode,gpu):
    dst=OUT/mode;dst.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-u','run_feng_ig_rescan.py','--config',f'/root/autodl-tmp/output/six_scene_extension_20261008/configs/{SEQ}.yaml',
         '--dataset',f'/root/autodl-tmp/3RScan/3RScan_Dynamic_Additional_20261008/{SEQ}/frames',
         '--instance_dir',f'/root/autodl-tmp/output/six_scene_extension_20261008/seg_result/{SEQ}',
         '--output_dir',str(dst),'--seq_name',SEQ,'--merge_selection','disjoint_pairs','--correction',mode]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu));start=time.time()
    with (dst/'run.log').open('x') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
    row=dict(mode=mode,gpu=gpu,command=cmd,returncode=r.returncode,seconds=time.time()-start)
    (dst/'run_result.json').write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
    return row
if __name__=='__main__':
    with ThreadPoolExecutor(2) as pool:rows=list(pool.map(lambda x:run(*x),[('detect',4),('delete',7)]))
    (OUT/'run_results.json').write_text(json.dumps(rows,indent=2))
    if any(x['returncode'] for x in rows):raise SystemExit(1)
