"""Run official code with an external invalid-depth guard and loop timing."""
import gc,json,os,random,runpy,sys,time
from pathlib import Path
import numpy as np
import torch,tqdm
repo=Path(os.environ['OASEG_BASELINE_ROOT']).resolve()
import subprocess
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip() == '507dfd5dc8fde286c1faab01c60d3c2e2068fe84', 'Expected pinned original baseline'
assert not subprocess.check_output(['git','-C',str(repo),'diff','--name-only','HEAD'],text=True).strip(), 'Baseline tracked source must be clean'
# Resolve caller paths before moving into the baseline checkout.
for flag in ['-c','-d','-i','-o']:
    if flag not in sys.argv:raise ValueError('Required baseline flag: '+flag)
    i=sys.argv.index(flag)+1;sys.argv[i]=str(Path(sys.argv[i]).resolve())
out_check=Path(sys.argv[sys.argv.index('-o')+1])/sys.argv[sys.argv.index('--seq_name')+1]
if out_check.exists() and any(out_check.iterdir()):raise FileExistsError(out_check)
out_check.parent.mkdir(parents=True,exist_ok=True)
os.chdir(repo);sys.path.insert(0,str(repo))
random.seed(27);np.random.seed(27);torch.manual_seed(27);torch.cuda.manual_seed_all(27)
real=tqdm.tqdm;loops=[]
class measured(real):
    def __iter__(self):
        start=time.perf_counter()
        try:
            for item in super().__iter__():yield item
        finally:loops.append(time.perf_counter()-start)
tqdm.tqdm=measured
sys.argv=[str(repo/'main.py')]+sys.argv[1:]
from baseline_empty_guard import run_guarded
start=time.perf_counter();namespace=run_guarded(repo/'main.py');torch.cuda.synchronize()
out=Path(sys.argv[sys.argv.index('-o')+1])/sys.argv[sys.argv.index('--seq_name')+1]
(out/'timing.json').write_text(json.dumps(dict(total_seconds=time.perf_counter()-start,
    main_loop_seconds=max(loops) if loops else None,seed=27,repository_entry_point_file_unchanged=True,
    runtime_input_guard='Skip only the exact Open3D no-block error when no depth is inside the existing configured range',
    skipped_invalid_depth_frames=namespace['__empty_depth_skipped'],
    original_main_sha256=namespace['__original_main_sha256']),indent=2))
# Release scene tensors while Open3D's CUDA cache is still alive. The original
# mapping entry point and all numerical operations above remain unchanged.
namespace.clear();gc.collect();torch.cuda.empty_cache()
import open3d as o3d
o3d.core.cuda.release_cache()
# This Open3D build aborts inside its process-global CUDA cache destructor even
# after explicit release. Exit only after the untouched main returned normally,
# CUDA synchronization succeeded, and all result files were closed and saved.
assert (out/'final.ply').stat().st_size>0 and (out/'ckpt_final.npz').stat().st_size>0
print('BASELINE_WRAPPER_COMPLETE: outputs saved; bypassing known Open3D shutdown destructor',flush=True)
sys.stdout.flush();sys.stderr.flush();os._exit(0)
