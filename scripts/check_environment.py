"""Check the installed CPU/GPU stack without starting an experiment."""
import argparse,json,sys,importlib.metadata

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cpu-only',action='store_true');a=p.parse_args()
    import numpy,scipy,open3d,cv2,yaml
    result={'python':sys.version,'numpy':numpy.__version__,'scipy':scipy.__version__,'open3d':open3d.__version__,'opencv':cv2.__version__}
    if not a.cpu_only:
        import torch,torch_scatter,MinkowskiEngine,pytorch3d
        from pytorch3d.ops import ball_query
        if not torch.cuda.is_available() or not open3d.core.cuda.is_available():raise RuntimeError('Both PyTorch and Open3D must support CUDA')
        if open3d.__version__!='0.18.0':raise RuntimeError('The compatibility guard requires Open3D 0.18.0')
        x=torch.ones(3,device='cuda');torch_scatter.scatter_add(x,torch.tensor([0,0,1],device='cuda'))
        y=open3d.core.Tensor([1.,2.],device=open3d.core.Device('CUDA:0'));assert y.cpu().numpy().sum()==3
        torch.cuda.synchronize()
        result.update(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),MinkowskiEngine=MinkowskiEngine.__version__,pytorch3d=pytorch3d.__version__)
    print(json.dumps(result,indent=2));print('ENVIRONMENT_CHECK_PASSED')
if __name__=='__main__':main()
