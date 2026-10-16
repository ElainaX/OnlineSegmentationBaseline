"""Preserve the sample axis for NumPy indexed by a one-element torch index."""
import inspect
import hashlib
import numpy as np
import torch

def preserve_index_axis(inds):
    if isinstance(inds, torch.Tensor) and inds.numel() == 1:
        print('FCGF_SINGLETON_INDEX_GUARD: preserving (1, C) sample axis', flush=True)
        return inds.detach().cpu().numpy().reshape(-1)
    return inds

def install():
    from third_party.FCGF.util import misc
    original = misc.extract_features_mdfy
    source = inspect.getsource(original)
    anchor = 'return_coords = xyz[inds]'
    assert source.count(anchor) == 1
    assert source.count('feats = feats[inds]') == 1
    # Match actual indentation, including repositories using four-space blocks.
    line = next(line for line in source.splitlines() if anchor in line)
    indent = line[:len(line)-len(line.lstrip())]
    patched = source.replace(line, indent+'inds = __preserve_index_axis(inds)\n'+line)
    misc.__dict__['__preserve_index_axis'] = preserve_index_axis
    exec(compile(patched, original.__code__.co_filename, 'exec'), misc.__dict__)
    print('FCGF_SINGLETON_GUARD_INSTALLED source_sha256='+hashlib.sha256(source.encode()).hexdigest(), flush=True)
    return original, misc.extract_features_mdfy
