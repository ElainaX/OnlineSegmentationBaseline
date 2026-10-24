"""Evaluate using the unchanged original Flat evaluator and fixed GT files."""
import argparse,json,sys
from pathlib import Path
from common import ROOT,AP_GT_SHA,CHANGE_GT_SHA,methods,sha

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path,help='Directory containing final.ply and ckpt_final.npz')
    p.add_argument('--method',required=True,choices=methods());a=p.parse_args();root=a.output.resolve()
    for rel,h in [('derived_gt_v1/run2_domain.npz',AP_GT_SHA),('derived_change_gt_v1/reference_domain.npz',CHANGE_GT_SHA)]:
        if sha(ROOT/'evaluation/flat'/rel)!=h:raise ValueError(f'Wrong fixed GT: {rel}')
    launch=root.parent/'launch.json'
    if launch.exists():
        rec=json.loads(launch.read_text())
        if rec['method']!=a.method:raise ValueError('Requested method differs from mapper launch')
        if rec['status']!='complete' or rec.get('smoke_frames'):raise ValueError('Only successful full-sequence outputs may enter the benchmark table')
    sys.path.insert(0,str(ROOT/'evaluation/flat'))
    from evaluate_flat_outputs import evaluate
    result=evaluate(root,methods()[a.method]['deletion_eval'])
    if result['deletion']['status']!='available_on_observed_domain':raise RuntimeError(result['deletion'])
    result['ablation_method']=a.method
    (root/'flat_metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    a=result['AP']['primary'];print(json.dumps({'AP_percent':a['AP']*100,'AP50_percent':a['AP50']*100,'AP25_percent':a['AP25']*100,'D_percent':result['deletion']['D_observed']*100,'GAP_percent':a['GAP_observed']*100}))
if __name__=='__main__':main()
