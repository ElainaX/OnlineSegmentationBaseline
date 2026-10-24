"""Install verified external weights and fixed evaluation domains."""
import argparse,shutil,json
from pathlib import Path
from common import ROOT,FCGF_PATH,FCGF_SHA,AP_GT_SHA,CHANGE_GT_SHA,sha

def install(source,dest,expected):
    source=Path(source).resolve()
    if sha(source)!=expected:raise ValueError(f'Wrong asset SHA256: {source}')
    if dest.exists():
        if sha(dest)!=expected:raise FileExistsError(f'Different asset already exists: {dest}')
        return
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fcgf',type=Path);p.add_argument('--ap-gt',type=Path);p.add_argument('--change-gt',type=Path)
    a=p.parse_args()
    if not any([a.fcgf,a.ap_gt,a.change_gt]):p.error('Provide at least one asset')
    if a.fcgf:install(a.fcgf,FCGF_PATH,FCGF_SHA)
    if a.ap_gt:install(a.ap_gt,ROOT/'evaluation/flat/derived_gt_v1/run2_domain.npz',AP_GT_SHA)
    if a.change_gt:install(a.change_gt,ROOT/'evaluation/flat/derived_change_gt_v1/reference_domain.npz',CHANGE_GT_SHA)
    print('ASSET_HASHES_VERIFIED')
if __name__=='__main__':main()
