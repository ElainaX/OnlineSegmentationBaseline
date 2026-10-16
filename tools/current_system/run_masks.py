"""External zero-instance serialization guard; original inference and thresholds unchanged."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'third_party/detectron2/projects/CropFormer/demo_cropformer/mask_predict_single_seq_w_semantic.py'

def empty_or_stack(features,model):
    import torch
    if features:return torch.stack(features,dim=0)
    return torch.empty((0,model.visual.output_dim),dtype=torch.float32,device=next(model.parameters()).device)

def resume_complete_frames(dataset):
    import cv2,torch
    remaining=[];skipped=[]
    for path in dataset.image_list:
        stem=Path(path).stem
        mask=Path(dataset.output_mask_dir)/(stem+'.png')
        color=Path(dataset.output_mask_color_dir)/(stem+'.png')
        features=Path(dataset.output_mask_feature_dir)/(stem+'.pt')
        if all(p.is_file() for p in [mask,color,features]):
            im=cv2.imread(str(mask),-1);co=cv2.imread(str(color),-1);fe=torch.load(features,map_location='cpu')
            assert im is not None and im.shape==(dataset.dst_h,dataset.dst_w)
            assert co is not None and co.shape==(dataset.dst_h,dataset.dst_w,3)
            assert torch.is_tensor(fe) and fe.ndim==2 and torch.isfinite(fe).all() and fe.shape[0]>=int(im.max())
            skipped.append(stem)
        else:remaining.append(path)
    dataset.image_list=remaining
    print('RESUME_COMPLETE_MASK_FRAMES',json.dumps(skipped),flush=True)

def patched_source(source):
    changes={
      'rgb_value_list = get_new_pallete(ranks.shape[0])':'rgb_value_list = get_new_pallete(ranks.shape[0]) if ranks.shape[0] else []',
      'mask_visual_embeddings = torch.stack(mask_visual_embedding_list, dim=0)':'mask_visual_embeddings = _empty_or_stack(mask_visual_embedding_list, model)',
      '    my_dataloader = DataLoader(my_dataset)':'    _resume_complete_frames(my_dataset)\n    my_dataloader = DataLoader(my_dataset)',
    }
    for old,new in changes.items():
        assert source.count(old)==1,old
        source=source.replace(old,new)
    return source

if __name__=='__main__':
    sys.path.insert(0,str(P.parent));sys.argv[0]=str(P)
    exec(compile(patched_source(P.read_text()),str(P),'exec'),dict(__name__='__main__',__file__=str(P),_empty_or_stack=empty_or_stack,_resume_complete_frames=resume_complete_frames))
