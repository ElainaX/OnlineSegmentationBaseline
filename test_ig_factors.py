"""CPU equivalence checks against the existing Carving-AABB equations."""
import ast
from pathlib import Path
from types import SimpleNamespace
import torch
from submap_disjoint.carved_pair_scores import adjust_carved_pair_scores
from experiments.baseline_aabb_soft_v2.aabb_soft import soft_similarity
from depth_mask_refinement import shrink_segmentation_by_depth

torch.manual_seed(27)
for trial in range(200):
    n=32
    lower=torch.rand(n,3)*2
    boxes=torch.stack((lower,lower+torch.rand(n,3)*2),1)
    if trial%2==0: boxes[0,1,2]=boxes[0,0,2]
    valid=torch.rand(n)>.1
    c=torch.rand(n,n)
    b=torch.rand(n,n)*.6+2
    b=(b+b.T)/2
    threshold=2.3; overlap=.4
    dense,stats=soft_similarity(b,c,boxes,valid,threshold,overlap,.15,.10)
    ij=torch.tril(torch.ones(n,n,dtype=torch.bool),diagonal=-1).nonzero()
    sparse,sstats=adjust_carved_pair_scores(b[ij[:,0],ij[:,1]],ij,
        c[ij[:,0],ij[:,1]],c[ij[:,1],ij[:,0]],boxes,valid,threshold,overlap,.15,.10)
    torch.testing.assert_close(sparse,dense[ij[:,0],ij[:,1]],rtol=0,atol=0)
    for key in ['eligible_pairs','promoted_pairs','demoted_pairs']:
        assert sstats[key]==stats[key],(trial,key,sstats,stats)
    unchanged,_=adjust_carved_pair_scores(b[ij[:,0],ij[:,1]],ij,
        c[ij[:,0],ij[:,1]],c[ij[:,1],ij[:,0]],boxes,valid,threshold,overlap,0,.10)
    assert torch.equal(unchanged,b[ij[:,0],ij[:,1]])

def load_method(path,name):
    tree=ast.parse(Path(path).read_text())
    func=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    env={'torch':torch}
    exec(compile(ast.Module(body=[func],type_ignores=[]),path,'exec'),env)
    return env[name]

old=load_method('experiments/run_disjoint_carving_aabb.py','_make_carved_boxes')
new=load_method('submap_disjoint/factor_owner.py','_make_carved_boxes')
for trial in range(20):
    points={i:torch.rand(100,3)*2 for i in range(4)}
    points[0]=torch.zeros(0,3)
    cfg={'scene':{'voxel_size':.025}}
    params=dict(c_mask_num=4,device='cpu',latest_depth=torch.ones(32,32),
        latest_pose=torch.eye(4),intrinsics=torch.tensor([[15.,0,16.],[0,15.,16.],[0,0,1.]]),
        cfg=cfg,mask_voxels_coords=points,_carve_frame=trial)
    a=SimpleNamespace(**params,carve_stats=[]);b=SimpleNamespace(**params,carve_stats=[])
    before={i:x.clone() for i,x in points.items()}
    torch.testing.assert_close(old(a),new(b),rtol=0,atol=0)
    assert a.carve_stats==b.carve_stats
    assert all(torch.equal(before[i],points[i]) for i in points)

cfg={'seg':{'depth_shrink':dict(enabled=True,front_quantile=.2,margin_m=.08,
    min_pixels=25,min_keep_ratio=.7,remove_unknown=False)}}
seg=torch.ones(10,10,dtype=torch.long)
depth=torch.ones(10,10);depth[-2:]=2
out,stats=shrink_segmentation_by_depth(seg,depth,cfg)
assert stats['removed_pixels']==20 and int((out==1).sum())==80
depth[-4:]=2
out,stats=shrink_segmentation_by_depth(seg,depth,cfg)
assert torch.equal(out,seg) and stats['removed_pixels']==0
print('PASS: 200 sparse/dense score and gate comparisons; 20 carving parity and map immutability checks; keep=0.70 shrink/fallback cases.')
