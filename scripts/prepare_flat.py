"""Prepare the official Flat RGB-D stream only; never consume GT labels."""
import argparse,json,shutil,tempfile,zipfile,re
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from common import require_empty,sha

def prepare(raw,output):
    raw=Path(raw);output=Path(output)
    intrinsic=(raw/'Intrinsics.txt').read_text()
    expected={'Res_x':640,'Res_y':480,'u':320,'v':240,'f_x':320,'f_y':320}
    for name,value in expected.items():
        match=re.search(r'^'+re.escape(name)+r':\s*([\d.]+)',intrinsic,re.M)
        if not match or float(match.group(1))!=value:raise ValueError(f'Unexpected official Flat intrinsics: {name}')
    require_empty(output)
    for sub in ['color','depth','pose','intrinsic']:(output/sub).mkdir()
    frame_index=[];index=0
    for run,count in [('run1',400),('run2',362)]:
        stems=sorted((raw/run).glob('*_color.png'))
        if len(stems)!=count:raise ValueError(f'{run}: expected {count} RGB frames, got {len(stems)}')
        for j,color in enumerate(stems):
            stem=color.name.removesuffix('_color.png')
            if int(stem)!=j:raise ValueError('Source frame IDs must be contiguous from zero')
            rgb=np.asarray(Image.open(color))
            depth=np.asarray(Image.open(raw/run/f'{stem}_depth.tiff'),dtype=np.float64)
            pose_path=raw/run/f'{stem}_pose.txt';pose=np.loadtxt(pose_path)
            if rgb.shape[:2]!=(480,640) or depth.shape!=(480,640):raise ValueError('Expected 640x480 images')
            if pose.shape!=(4,4) or not np.isfinite(pose).all() or not np.allclose(pose[3],[0,0,0,1]):raise ValueError('Invalid camera-to-world pose')
            # Preserve PNG bytes under the .jpg filename required by the loader.
            shutil.copyfile(color,output/'color'/f'{index}.jpg')
            mm=np.rint(np.where(np.isfinite(depth),depth,0)*1000.)
            valid=(mm>0)&(mm<=65535)
            image=np.where(valid,mm,0).astype(np.uint16)
            if not cv2.imwrite(str(output/'depth'/f'{index}.png'),image):raise OSError('Depth write failed')
            shutil.copyfile(pose_path,output/'pose'/f'{index}.txt')
            frame_index.append({'frame':index,'run':run,'source_frame':j,'valid_depth_pixels':int(((depth>.1)&(depth<5)&np.isfinite(depth)).sum())})
            index+=1
    (output/'intrinsic/intrinsic_depth.txt').write_text('320 0 320 0\n0 320 240 0\n0 0 1 0\n0 0 0 1\n')
    boundary={'reference_frame_count':400,'rescan_frame_count':362,'total_frame_count':762,'rescan_start_frame':400,'stream_order':['run1','run2'],'pose_convention':'camera_to_world','coordinate_system':'Flat source world','frame_index':frame_index}
    (output/'session_boundary.json').write_text(json.dumps(boundary,indent=2))
    print(json.dumps({'frames':762,'fusion_frames':191,'output':str(output.resolve()),'GT_used':False}))

def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--raw',type=Path,help='Extracted flat_dataset containing run1/ and run2/')
    g.add_argument('--archive',type=Path,help='Official flat_dataset.zip')
    p.add_argument('--output',required=True,type=Path,help='New .../flat_run1_run2/frames directory')
    a=p.parse_args()
    if a.raw:prepare(a.raw,a.output);return
    with tempfile.TemporaryDirectory(prefix='flat_raw_') as d:
        root=Path(d).resolve()
        with zipfile.ZipFile(a.archive) as z:
            for m in z.infolist():
                dest=(root/m.filename).resolve()
                if not dest.is_relative_to(root):raise ValueError('Unsafe archive path')
            z.extractall(root)
        candidates=[q.parent for q in root.rglob('run1') if q.is_dir() and (q.parent/'run2').is_dir()]
        if len(candidates)!=1:raise ValueError('Cannot locate unique flat_dataset root')
        prepare(candidates[0],a.output)
        (a.output.parent/'archive_sha256.txt').write_text(sha(a.archive)+'\n')

if __name__=='__main__':main()
