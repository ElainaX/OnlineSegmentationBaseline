"""Optional CropFormer + CLIP prediction. Use shared cached masks for exact comparisons."""
import argparse,os,sys
from pathlib import Path
from common import ROOT

def empty_or_stack(features,model):
    import torch
    if features:return torch.stack(features,dim=0)
    return torch.empty((0,model.visual.output_dim),dtype=torch.float32,device=next(model.parameters()).device)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cropformer-root',required=True,type=Path,help='External CropFormer directory containing mask2former/, configs/, demo_cropformer/')
    p.add_argument('--cropformer-weights',required=True,type=Path);p.add_argument('--clip-weights',required=True,type=Path)
    p.add_argument('--data-root',required=True,type=Path,help='Parent of flat_run1_run2/frames/color')
    p.add_argument('--output-root',required=True,type=Path);p.add_argument('--seq-name',default='flat_run1_run2');p.add_argument('--gpu',default='0')
    p.add_argument('--limit-frames',type=int,default=0,help='Smoke only: limit selected RGB frames; 0 processes all frames')
    a=p.parse_args();crop=a.cropformer_root.resolve()
    config=crop/'configs/entityv2/entity_segmentation/mask2former_hornet_3x.yaml'
    for q in [config,crop/'demo_cropformer/predictor.py',a.cropformer_weights,a.clip_weights]:
        if not q.is_file():raise FileNotFoundError(q)
    output=a.output_root.resolve()/a.seq_name
    if output.exists() and any(output.iterdir()):raise FileExistsError('Use a new mask output directory')
    source=ROOT/'preprocess/masks/mask_predict_single_seq_w_semantic.py'
    text=source.read_text()
    changes={
        'rgb_value_list = get_new_pallete(ranks.shape[0])':'rgb_value_list = get_new_pallete(ranks.shape[0]) if ranks.shape[0] else []',
        'mask_visual_embeddings = torch.stack(mask_visual_embedding_list, dim=0)':'mask_visual_embeddings = _empty_or_stack(mask_visual_embedding_list, model)',
        '    my_dataloader = DataLoader(my_dataset)':'    if _limit_frames: my_dataset.image_list = my_dataset.image_list[:_limit_frames]\n    my_dataloader = DataLoader(my_dataset)',
    }
    for old,new in changes.items():
        if text.count(old)!=1:raise RuntimeError(f'Unexpected prediction source: {old}')
        text=text.replace(old,new)
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    sys.path[:0]=[str(source.parent),str(crop/'demo_cropformer'),str(crop)]
    sys.argv=[str(source),'--config-file',str(config),'--root',str(a.data_root.resolve()),'--seq_name',a.seq_name,'--output_root',str(a.output_root.resolve()),'--seg_interval','4','--dst_h','480','--dst_w','640','--pretrained_path',str(a.clip_weights.resolve()),'--opts','MODEL.WEIGHTS',str(a.cropformer_weights.resolve())]
    exec(compile(text,str(source),'exec'),dict(__name__='__main__',__file__=str(source),_empty_or_stack=empty_or_stack,_limit_frames=a.limit_frames))
if __name__=='__main__':main()
