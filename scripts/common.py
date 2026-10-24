"""Shared paths and immutable file checks for the public command-line tools."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
AP_GT_SHA = '7a01a25821e8e9e94044efbbcd6584f8c8f4c33a46a53b7e11574f34a96ff8a9'
CHANGE_GT_SHA = 'ca1aeb0b319f993a0f36afbfc4106ed66383aadb2540819510ba003762aeb1c5'
FCGF_SHA = '5dfc8d30d08f915b61c133a02d4ac11c90de6eb028e73f273e9c8810f9362496'
FCGF_PATH = ROOT/'third_party/FCGF/model/ResUNetBN2C-16feat-3conv.pth'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def methods():
    return json.loads((ROOT/'configs/ablations.json').read_text())

def require_empty(path):
    path=Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'Use a new empty directory: {path}')
    path.mkdir(parents=True,exist_ok=True)

def check_layout(frames,masks):
    frames,masks=Path(frames),Path(masks)
    for i in range(762):
        for sub,suffix in [('color','.jpg'),('depth','.png'),('pose','.txt')]:
            if not (frames/sub/f'{i}{suffix}').is_file():
                raise FileNotFoundError(frames/sub/f'{i}{suffix}')
    for sub,suffix in [('color','.jpg'),('depth','.png'),('pose','.txt')]:
        if len(list((frames/sub).glob('*'+suffix)))!=762:
            raise ValueError(f'Expected exactly 762 {sub} files')
    for i in range(0,762,4):
        for sub,suffix in [('mask','.png'),('mask_embeddings','.pt')]:
            if not (masks/sub/f'{i}{suffix}').is_file():raise FileNotFoundError(masks/sub/f'{i}{suffix}')
    boundary=json.loads((frames/'session_boundary.json').read_text())
    if boundary['rescan_start_frame']!=400:raise ValueError('Expected run2 boundary at frame 400')
    if not (frames/'intrinsic/intrinsic_depth.txt').is_file():raise FileNotFoundError('Missing intrinsics')
    return {'source_frames':762,'fusion_frames':191,'rescan_start':400}
