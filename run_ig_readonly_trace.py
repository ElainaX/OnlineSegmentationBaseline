"""Diagnostic entry point: periodic in-process Python stacks, same IG algorithm."""
import faulthandler
import runpy

faulthandler.enable()
faulthandler.dump_traceback_later(60, repeat=True)
try:
    runpy.run_path('run_ig_readonly.py', run_name='__main__')
finally:
    faulthandler.cancel_dump_traceback_later()
