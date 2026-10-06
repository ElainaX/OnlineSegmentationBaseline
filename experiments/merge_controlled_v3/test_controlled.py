import torch
from run_controlled import score, prototype, camera_quality

x = torch.tensor([[0., .5, 1.]])
for mode in ('baseline','no_semantic','no_geometry','no_overlap_score'):
    assert torch.allclose(score(mode,x,x,x,True),3*x)
    assert torch.allclose(score(mode,x,x,x,False),2*x)
a = torch.tensor([[1.,0.],[1.,0.]])
p, fused = prototype(a,torch.ones(2),True)
assert fused and torch.allclose(p,torch.tensor([2.,0.]))
a = torch.eye(2)
p, fused = prototype(a,torch.tensor([1.,2.]),True)
assert not fused and torch.equal(p,torch.tensor([0.,1.]))
p, fused = prototype(a,torch.ones(2),False)
assert fused and torch.equal(p,torch.ones(2))
poses = torch.eye(4).repeat(3,1,1)
assert camera_quality(poses,2)==1.
poses[2,0,3]=.1
assert 0 < camera_quality(poses,2)<1
poses[2]=float('inf')
assert camera_quality(poses,2)==.01
print('score scale, prototype gating, stationary/moving/invalid pose checks passed')
