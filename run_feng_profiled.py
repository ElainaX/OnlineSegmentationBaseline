"""Read-only timing instrumentation for the existing experiment runner."""
import functools, json, os, runpy, sys, time, traceback
from pathlib import Path

out = Path(sys.argv[sys.argv.index('--output_dir')+1])/sys.argv[sys.argv.index('--seq_name')+1]
out.parent.mkdir(parents=True, exist_ok=True)
profile_path = out.parent/(out.name+'.profile.json')
stats, stack = {}, []

def save():
    p=profile_path.with_suffix('.tmp')
    p.write_text(json.dumps({'timing':'host wall, inclusive and exclusive nested spans; no extra GPU barriers', 'functions':stats},indent=2))
    p.replace(profile_path)

def wrap(obj,name,label=None):
    original=getattr(obj,name); label=label or obj.__name__+'.'+name
    @functools.wraps(original)
    def measured(*a,**kw):
        frame=[time.perf_counter(),0.];stack.append(frame)
        try:return original(*a,**kw)
        finally:
            dt=time.perf_counter()-frame[0];stack.pop()
            if stack:stack[-1][1]+=dt
            row=stats.setdefault(label,{'calls':0,'inclusive_seconds':0.,'exclusive_seconds':0.,'max_seconds':0.})
            row['calls']+=1;row['inclusive_seconds']+=dt;row['exclusive_seconds']+=dt-frame[1];row['max_seconds']=max(row['max_seconds'],dt)
            if not stack:save()
    setattr(obj,name,measured)

sys.path.append('third_party/FCGF')
import feng_ig_adapter as adapter
import feng_remove_detection as detector
from submap_disjoint.ig_correction import IGCorrection
for name in ['step','delete_candidate','save','finalize']:wrap(adapter.FengIGCorrection,name)
for name in ['raycast_history','lookup_voxels_safe']:wrap(adapter.GeometryView,name)
wrap(adapter.HashView,'query_active_instance_memberships')
wrap(adapter,'exclusive_cells')
wrap(IGCorrection,'erase')
for name in ['_step','_reference','_surface','_save_candidate','_dump_frame','_event','_summary','finalize']:wrap(detector.RemoveDetector,name)
for name in ['unambiguous_owners','project_reference','build_rgb_context','interior_mask','owner_interior','write_ply']:wrap(detector,name)
wrap(detector.Candidate,'observe')
code=0
try:
    runpy.run_path('run_instdel_multires.py',run_name='__main__')
except BaseException:
    traceback.print_exc();code=1
finally:
    save();sys.stdout.flush();sys.stderr.flush()
# Output metadata is written before returning; bypass known Open3D teardown crash.
os._exit(code)
