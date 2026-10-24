"""Run one frozen Flat method. This command starts the algorithm, not scoring."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from common import ROOT,FCGF_PATH,FCGF_SHA,methods,check_layout,sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=methods(),default='full')
    p.add_argument('--frames',required=True,type=Path);p.add_argument('--masks',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path,help='New output parent; final files are under flat_run1_run2/')
    p.add_argument('--gpu',default='0');p.add_argument('--smoke-frames',type=int,default=0)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();cfg=methods()[a.method];frames=a.frames.resolve();masks=a.masks.resolve();out=a.output.resolve()
    check_layout(frames,masks)
    if sha(FCGF_PATH)!=FCGF_SHA:raise ValueError('Run prepare_assets.py with the correct FCGF weight')
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'Use a new output directory: {out}')
    if a.smoke_frames and (a.method=='baseline' or not 1<=a.smoke_frames<=762):p.error('Smoke mode supports Full-family methods and 1..762 frames')
    command=[sys.executable,'-u',str(ROOT/'adapters'/cfg['entry'])]
    if a.method=='baseline':
        command+=['-c',str(ROOT/'configs/flat.yaml'),'-d',str(frames),'-i',str(masks),'-o',str(out),'--seq_name','flat_run1_run2']
    else:
        command+=['--config',str(ROOT/'configs/flat.yaml'),'--dataset',str(frames),'--instance_dir',str(masks),'--output_dir',str(out),'--seq_name','flat_run1_run2','--merge_selection','disjoint_pairs','--correction',cfg['correction'],'--global-voxel-m','0.025','--aabb','0','--depth-shrink','0','--detector-config',str(ROOT/'configs'/cfg['detector'])]
        if cfg['refine']:command+=['--observation-refine']
        if a.smoke_frames:command+=['--smoke-stop',str(a.smoke_frames)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONPATH=str(ROOT/'adapters')+os.pathsep+str(ROOT),FENG_VARIANT='fast',ABLATION_METHOD=cfg.get('env_method',''))
    if cfg.get('cuda_launch_blocking'):env['CUDA_LAUNCH_BLOCKING']='1'
    record={'method':a.method,'command':command,'settings':cfg,'smoke_frames':a.smoke_frames,'config_sha256':sha(ROOT/'configs/flat.yaml'),'gpu':a.gpu,'status':'planned','ground_truth_used_by_mapping':False}
    print(json.dumps(record,indent=2),flush=True)
    if a.dry_run:return
    out.mkdir(parents=True,exist_ok=True);record.update(status='running',started=time.time());(out/'launch.json').write_text(json.dumps(record,indent=2))
    with (out/'mapping.log').open('x') as log:
        run=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    record.update(status='complete' if run.returncode==0 else 'failed',returncode=run.returncode,finished=time.time())
    if run.returncode==0:
        for name in ['final.ply','ckpt_final.npz']:
            if not (out/'flat_run1_run2'/name).is_file():record['status']='failed';record['reason']='Missing final output'
    (out/'launch.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record),flush=True)
    if record['status']!='complete':raise SystemExit(run.returncode or 1)
if __name__=='__main__':main()
