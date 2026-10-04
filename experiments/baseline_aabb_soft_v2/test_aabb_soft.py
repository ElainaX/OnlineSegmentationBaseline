import torch
from aabb_soft import soft_similarity


def check(device):
    boxes = torch.tensor([[[0.,0.,0.],[1.,1.,1.]],
                          [[0.,0.,0.],[1.,1.,1.]],
                          [[2.,0.,0.],[3.,1.,1.]]],device=device)
    valid = torch.ones(3,dtype=torch.bool,device=device)
    base = torch.full((3,3),2.25,device=device)
    c = torch.full((3,3),.6,device=device)
    out, _ = soft_similarity(base,c,boxes,valid,2.3,.4,0,0.1)
    assert out is base  # zero-weight path is exactly the untouched baseline
    out, stats = soft_similarity(base,c,boxes,valid,2.3,.4,.1,.1)
    assert out[0,1]>2.3 and out[0,2]<base[0,2]
    assert torch.equal(out,out.T) and torch.equal(out.diag(),base.diag())
    # Translation and metre-to-centimetre conversion preserve the score.
    moved,_=soft_similarity(base,c,(boxes+123)*100,valid,2.3,.4,.1,.1)
    assert torch.allclose(out,moved,atol=1e-5)
    # No AABB match can invent an eligible pair or touch confident decisions.
    c[0,1]=c[1,0]=0
    base[1,2]=base[2,1]=2.6
    out,_=soft_similarity(base,c,boxes,valid,2.3,.4,.1,.1)
    assert out[0,1]==base[0,1] and out[1,2]==base[1,2]
    valid[0]=False
    out,_=soft_similarity(base,c,boxes,valid,2.3,.4,.1,.1)
    assert torch.equal(out,base)


if __name__=='__main__':
    check('cpu')
    if torch.cuda.is_available(): check('cuda')
    print('PASS: zero-weight identity, candidate preservation, strong-score fallback, geometry invariance, invalid-box fallback, CPU/CUDA')
