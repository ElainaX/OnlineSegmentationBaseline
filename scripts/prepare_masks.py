"""Copy and validate the shared predicted masks/CLIP features, without GT."""
import argparse,json,shutil
from pathlib import Path
import cv2,numpy as np,torch
from common import require_empty,sha

def validate(root):
    records={}
    for i in range(0,762,4):
        mask=root/'mask'/f'{i}.png';feat=root/'mask_embeddings'/f'{i}.pt'
        image=cv2.imread(str(mask),cv2.IMREAD_UNCHANGED)
        if image is None or image.shape!=(480,640):raise ValueError(f'Invalid mask: {mask}')
        f=torch.load(feat,map_location='cpu',weights_only=True)
        if not torch.is_tensor(f) or f.ndim!=2 or f.shape[1]!=1024 or not torch.isfinite(f).all() or f.shape[0]<int(image.max()):raise ValueError(f'Invalid features: {feat}')
        records[str(i)]={'mask_sha256':sha(mask),'feature_sha256':sha(feat),'instances':int(f.shape[0])}
    return records

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path)
    p.add_argument('--output',type=Path,help='New directory; omit to validate in place')
    a=p.parse_args();records=validate(a.source)
    if a.output:
        require_empty(a.output)
        for sub,suffix in [('mask','.png'),('mask_embeddings','.pt')]:
            (a.output/sub).mkdir()
            for i in range(0,762,4):shutil.copyfile(a.source/sub/f'{i}{suffix}',a.output/sub/f'{i}{suffix}')
        (a.output/'mask_audit.json').write_text(json.dumps({'frames':191,'GT_used':False,'files':records},indent=2))
    print('SHARED_MASKS_VALIDATED: 191 frames, 640x480 labels, 1024-dimensional features')
if __name__=='__main__':main()
