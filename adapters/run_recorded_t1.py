import os,sys,runpy,faulthandler
from pathlib import Path
out=Path(sys.argv[sys.argv.index('--output_dir')+1])/sys.argv[sys.argv.index('--seq_name')+1]
trace_stream=(out.parent/(out.name+'.stacks.log')).open('a')
faulthandler.enable(file=trace_stream)
faulthandler.dump_traceback_later(120,repeat=True,file=trace_stream)
sys.path.insert(0,os.getcwd())
sys.path.append('third_party/FCGF')
sys.path.append('third_party/FCGF')
from fcgf_singleton_guard import install as install_fcgf_singleton_guard
install_fcgf_singleton_guard()
from raycast_bounds_guard import install as install_raycast_bounds_guard
install_raycast_bounds_guard(out)
from immediate_confirmation import install as install_hybrid
install_hybrid(out)
from record_erasure import install
install()
runpy.run_path(str(Path(__file__).with_name('run_feng_optimized.py')),run_name='__main__')
