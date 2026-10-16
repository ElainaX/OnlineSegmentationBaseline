"""Run the original OnlineAnySeg Baseline at a chosen global TSDF resolution.

The frame list is shared with the IGSubmap-InstDel experiment.  This keeps the
comparison about reconstruction/association rather than mask availability.
Feng's contradiction detector is attached only as an optional deletion layer;
the underlying instance association remains the original Scene_rep code.
"""
import argparse, json, os, subprocess, sys, threading, time, hashlib
from pathlib import Path

import numpy as np
import torch

ORIGINAL = '/root/OnlineAnySegBaseline_original'
sys.path.insert(0, ORIGINAL)
sys.path.append(os.path.join(ORIGINAL, 'third_party', 'FCGF'))
from Dataset.dataset import get_dataset
from tool.config import load_config
from Scene_rep import Scene_rep
from feng_ig_adapter import FengIGCorrection


class GpuPeakMonitor:
    """Conservative whole-GPU memory peak, including Open3D allocations."""
    def __init__(self):
        ids = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')
        self.gpu = ids[0].strip() if ids else '0'
        self.peak_mib = 0
        self.samples = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            try:
                out = subprocess.check_output([
                    'nvidia-smi', '-i', self.gpu,
                    '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                    stderr=subprocess.DEVNULL, text=True, timeout=2)
                self.peak_mib = max(self.peak_mib, int(float(out.strip().splitlines()[0])))
                self.samples += 1
            except Exception:
                pass
            self.stop.wait(0.5)

    def __enter__(self):
        self.thread.start(); return self

    def __exit__(self, *_):
        self.stop.set(); self.thread.join(timeout=3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--instance_dir', required=True)
    p.add_argument('--seq_name', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--frame-ids-file', required=True)
    p.add_argument('--global-voxel-m', type=float, required=True)
    p.add_argument('--correction', choices=['off', 'delete'], default='delete')
    p.add_argument('--detector-config', default='feng_detector_config.json')
    a = p.parse_args()
    # The original extractor stores its checkpoint path relative to the
    # original repository.  Make this independent of the caller's cwd.
    os.chdir(ORIGINAL)
    out = Path(a.output_dir) / a.seq_name
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('output exists: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)
    frame_ids = [int(x) for x in json.loads(Path(a.frame_ids_file).read_text())]
    cfg = load_config(a.config)
    cfg['scene']['voxel_size'] = float(a.global_voxel_m)
    args = argparse.Namespace(output_dir=a.output_dir, seq_name=a.seq_name,
                              device='cuda:0')
    start = time.perf_counter()
    data = get_dataset(a.dataset, a.instance_dir, cfg, 'cuda:0')
    owner = Scene_rep(cfg, args, data, 'cuda:0')
    # FengIGCorrection's eraser invalidates both owner and owner.geometry;
    # original Scene_rep owns the geometry directly, so expose that alias.
    owner.geometry = owner
    correction = None
    if a.correction == 'delete':
        detector_path = Path(a.detector_config)
        if not detector_path.is_absolute():
            detector_path = Path(__file__).resolve().parent / detector_path
        correction = FengIGCorrection(owner, out / 'correction', enabled=True,
                                       options=json.loads(detector_path.read_text()))
    last_pose = torch.eye(4, device='cuda:0')
    integrated = 0
    observed = 0
    last_frame = frame_ids[-1]
    with GpuPeakMonitor() as mem:
        torch.cuda.reset_peak_memory_stats()
        for frame_id in frame_ids:
            rgb, depth, pose, seg, features, flag = data[frame_id]
            valid_pose = bool(torch.isfinite(pose).all())
            if not valid_pose:
                pose = last_pose
            else:
                last_pose = pose
            if correction is not None:
                if not correction.initialized:
                    correction.freeze(frame_id)
                if valid_pose:
                    correction.step(frame_id, rgb.cpu().numpy(), depth.cpu().numpy(),
                                    data.cam_intrinsic.cpu().numpy(), pose.cpu().numpy())
            if frame_id % cfg['mapping']['keyframe_freq'] != 0:
                continue
            with torch.no_grad():
                blocks, _ = owner.integrate_frame(frame_id, rgb, depth, torch.inverse(pose))
                integrated += 1
                if bool(flag) and frame_id % cfg['seg']['seg_add_interval'] == 0:
                    owner.insert_seg_frame(frame_id, rgb, depth, pose, blocks, seg, features)
                    observed += 1
                    if frame_id > 0 and frame_id % owner.merge_frame_interval == 0:
                        owner.update_masks(frame_id)
        torch.cuda.synchronize()
        ckpt_dir = out / ('ckpt_%d' % last_frame)
        ckpt_dir.mkdir(exist_ok=True)
        owner.save_ckpt(last_frame, str(ckpt_dir), filter_flag=True,
                        ckpt_path=str(out / 'ckpt_final.npz'))
        owner.save_merging_result(last_frame, reextract=False)
        owner.save_pc(owner.points, owner.colors, str(out / 'final.ply'))
        if correction is not None:
            correction.finalize()
        torch.cuda.synchronize()
    summary = None
    if correction is not None and (out / 'correction' / 'summary.json').exists():
        summary = json.loads((out / 'correction' / 'summary.json').read_text())
    meta = dict(method='Baseline-original', correction_mode=a.correction,
                correction_enabled=correction is not None, deletion_enabled=correction is not None,
                global_voxel_m=a.global_voxel_m, global_voxel_mm=a.global_voxel_m * 1000,
                processed_frame_ids=frame_ids, processed_frame_count=len(frame_ids),
                raw_frame_count=len(data), integrated_frames=integrated,
                segmented_frames=observed, gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                gpu_peak_memory_mib=mem.peak_mib, gpu_memory_samples=mem.samples,
                correction_summary=summary, total_seconds=time.perf_counter() - start,
                config_sha256=hashlib.sha256(Path(a.config).read_bytes()).hexdigest())
    (out / 'run_metadata.json').write_text(json.dumps(meta, indent=2))
    print('COMPLETE', json.dumps({'method': meta['method'], 'total_seconds': meta['total_seconds'],
                                  'gpu_peak_memory_mib': meta['gpu_peak_memory_mib'],
                                  'deletion_enabled': meta['deletion_enabled']}), flush=True)


if __name__ == '__main__':
    main()
