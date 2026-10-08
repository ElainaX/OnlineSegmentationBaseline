"""Static negative controls for raw RGB-D surface memory; no instance GT inputs."""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np,yaml
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from submap_disjoint.surface_bank import SurfaceBank
from submap_disjoint.readonly_detection_core import interior_mask

parser=argparse.ArgumentParser();parser.add_argument('--output-dir',required=True)
parser.add_argument('--surface-support',action='store_true');args=parser.parse_args()
OUT=Path(args.output_dir);OUT.mkdir(exist_ok=False)
config_path=Path('/root/autodl-tmp/output/a100_baseline_default_s5_20261007/scannet_default.yaml')
config=yaml.safe_load(config_path.read_text());cam=config['cam']
cfg=yaml.safe_load(Path('config/ig_readonly_onlineSeg.yaml').read_text())['remove_detection']
results=[]
for seq in ['scene0050_00','scene0549_00']:
    data=Path('/root/autodl-tmp/ScanNet_subset_onlineanyseg/data')/seq;out=OUT/seq;out.mkdir()
    k=np.loadtxt(data/'intrinsic/intrinsic_depth.txt')[:3,:3].astype(np.float32)
    hc,wc=cam['h_crop'],cam['w_crop'];k[0,2]-=wc;k[1,2]-=hc
    files={int(p.stem):p for p in (data/'depth').glob('*.png') if p.stem.isdigit()}
    selected=[f for f in sorted(files) if f%10==0]
    bank=SurfaceBank(config['scene']['voxel_size'],cfg,require_color=True,view_gate='ray',surface_support=args.surface_support)
    stats=[];start=time.perf_counter()
    for i,f in enumerate(selected):
        depth=cv2.imread(str(files[f]),cv2.IMREAD_UNCHANGED).astype(np.float32)/cam['depth_scale']
        color=cv2.imread(str(data/'color'/f'{f}.jpg'))
        if color is None:raise FileNotFoundError(data/'color'/f'{f}.jpg')
        rgb=cv2.cvtColor(color,cv2.COLOR_BGR2RGB)
        if rgb.shape[:2]!=depth.shape:rgb=cv2.resize(rgb,(depth.shape[1],depth.shape[0]),interpolation=cv2.INTER_NEAREST)
        sy=slice(hc,-hc if hc else None);sx=slice(wc,-wc if wc else None)
        depth=depth[sy,sx];rgb=rgb[sy,sx]
        valid=np.isfinite(depth)&(depth>cam['depth_near'])&(depth<cam['depth_far']);depth[~valid]=0
        inner=interior_mask(depth,valid,cfg['edge_threshold_m'],cfg['boundary_margin_px'])
        pose=np.loadtxt(data/'pose'/f'{f}.txt').astype(np.float32)
        if not np.isfinite(pose).all():continue
        t=time.perf_counter();row=bank.step(f,pose,k,depth,inner,cam['depth_near'],cam['depth_far'],rgb)
        row['seconds']=time.perf_counter()-t;stats.append(row)
        if i%100==0:print(seq,i,len(selected),'points',len(bank.points),flush=True)
    np.savez_compressed(out/'bank.npz',**bank.arrays())
    points=bank.points[bank.active]
    if len(points):
        edges=cKDTree(points).query_pairs(3*bank.voxel_size,output_type='ndarray')
        graph=coo_matrix((np.ones(len(edges),np.uint8),(edges[:,0],edges[:,1])),shape=(len(points),len(points)))
        _,labels=connected_components(graph,directed=False);sizes=np.bincount(labels)
    else:sizes=np.empty(0,int)
    result=dict(seq=seq,selected_frames=len(selected),valid_frames=len(stats),frame_interval=10,
        voxel_size=bank.voxel_size,config_path=str(config_path),cam=cam,cfg=cfg,GT_used=False,surface_support=args.surface_support,
        confirmed_points=len(points),regions20=int((sizes>=20).sum()),regions100=int((sizes>=100).sum()),
        wall_seconds=time.perf_counter()-start,mean_step_seconds=float(np.mean([r['seconds'] for r in stats])),
        max_active_points=max(r['confirmed_points'] for r in stats),frames=stats,
        note='Static control: alarms indicate geometric/appearance inconsistency, not known object disappearance.')
    (out/'summary.json').write_text(json.dumps(result,indent=2));results.append({k:v for k,v in result.items() if k!='frames'})
    print(seq,{k:result[k] for k in ['confirmed_points','regions20','regions100','wall_seconds']},flush=True)
(OUT/'comparison.json').write_text(json.dumps(dict(completed=True,scenes=results),indent=2))
