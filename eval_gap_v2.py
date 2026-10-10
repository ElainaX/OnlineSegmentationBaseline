"""Original AP plus event-grounded, bounded non-negative deletion gain.

G-AP_t = AP_t + (1 - AP_t) * D.  No retention re-scoring.
D uses fixed GT removed-object targets, actual executed deletion regions,
one-to-one size-weighted IoU and unmatched deletion support in its denominator.
Archived deletion sources must remain absent in the final exported geometry.
All lengths are metres. This is a custom bonus metric, not official AP.
"""
from pathlib import Path
import json, hashlib
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

VOXEL=.025
TOLERANCE=.05

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def cloud(path):
 import open3d as o3d
 p=np.asarray(o3d.io.read_point_cloud(str(path)).points)
 if not np.isfinite(p).all():raise ValueError('Nonfinite geometry')
 return p

def downsample(points,labels=None,voxel=VOXEL):
 p=np.asarray(points,float).reshape(-1,3)
 if not len(p):return p,np.empty(0,int)
 keys=np.floor(p/voxel).astype(np.int64)
 if labels is not None:keys=np.c_[labels,keys]
 _,first,inverse=np.unique(keys,axis=0,return_index=True,return_inverse=True)
 n=np.bincount(inverse)
 centers=np.column_stack([np.bincount(inverse,weights=p[:,i])/n for i in range(3)])
 return centers,first

def nearest(source,target):
 if not len(target):return np.full(len(source),np.inf),np.full(len(source),-1,int)
 return cKDTree(target).query(source)

def combine(standard_ap,deletion_quality):
 ap=float(standard_ap); d=float(deletion_quality)
 if not np.isfinite([ap,d]).all() or not 0<=ap<=1 or not 0<=d<=1:raise ValueError('Scores must lie in [0,1]')
 bonus=(1-ap)*d
 return dict(AP=ap,D=d,bonus=bonus,G_AP=ap+bonus)

def score_deletions(iou,gt_sizes,pred_sizes):
 """Only executed deletion predictions enter. Ordinary segmentation is untouched.

 Fixed target weight prevents tiny selective deletions receiving full credit.
 Unattempted removed targets contribute no numerator and never subtract AP.
 Wrong deletion support dilutes D, but cannot make the gain negative.
 """
 iou=np.asarray(iou,float);w=np.asarray(gt_sizes,float);p=np.asarray(pred_sizes,float)
 if iou.shape!=(len(p),len(w)):raise ValueError('Shape mismatch')
 if not np.isfinite(iou).all() or np.any((iou<0)|(iou>1)) or np.any(w<=0) or np.any(p<=0):raise ValueError('Invalid input')
 matched={}
 if iou.size:
  rows,cols=linear_sum_assignment(-iou*w[None,:])
  matched={int(r):int(c) for r,c in zip(rows,cols) if iou[r,c]>0}
 extra=[i for i in range(len(p)) if i not in matched]
 numerator=float(sum(w[c]*iou[r,c] for r,c in matched.items()))
 denominator=float(w.sum()+p[extra].sum())
 return dict(score=numerator/denominator if denominator else 0.,numerator=numerator,
   denominator=denominator,target_weight=float(w.sum()),unmatched_deletion_weight=float(p[extra].sum()),
   matches=matched,unmatched_predictions=extra)

def raw_reference(path,expected_points):
 """Read original object IDs including categories excluded from standard AP."""
 with Path(path).open() as f:
  if f.readline().strip()!='ply' or f.readline().strip()!='format ascii 1.0':raise ValueError('Expected ASCII annotation PLY')
  props=[];vertex=False
  for line in f:
   x=line.split()
   if x[:2]==['element','vertex']:n=int(x[2]);vertex=True
   elif x[:1]==['element']:vertex=False
   elif vertex and x[0]=='property':props.append(x[-1])
   if x[0]=='end_header':break
  data=np.loadtxt(f,max_rows=n)
 xyz=data[:,[props.index(k) for k in ['x','y','z']]]
 if xyz.shape!=expected_points.shape or not np.allclose(xyz,expected_points,atol=1e-6):raise ValueError('Raw annotation order/alignment mismatch')
 return data[:,props.index('objectId')].astype(int),data[:,props.index('NYU40')].astype(int)

def target_domain(initial,reference,labels,current,removed_ids):
 ref,first=downsample(reference,labels);lab=np.asarray(labels)[first]
 initial,_=downsample(initial)
 built=nearest(ref,initial)[0]<=TOLERANCE
 reused=nearest(ref,current)[0]<=TOLERANCE
 target=(lab>0)&np.isin(lab,list(removed_ids))&built&~reused
 ids=np.unique(lab[target]);sizes=np.array([(target&(lab==g)).sum() for g in ids],int)
 return dict(points=ref,labels=lab,target=target,ids=ids,sizes=sizes,
  per_GT=[dict(GT_id=int(g),reference_samples=int((lab==g).sum()),built_samples=int(((lab==g)&built).sum()),
    reused_samples=int(((lab==g)&built&reused).sum()),target_samples=int(((lab==g)&target).sum())) for g in sorted(removed_ids)])

def executed_regions(method,root,initial,cut):
 """Return authoritative action records, never infer deletion from off/on runs."""
 rows=[];input_files=[]
 if method=='IGSubmap_delete':
  p=root/'correction/summary.json';s=json.loads(p.read_text());input_files.append(p)
  files=sorted((root/'correction/erasure_archive').glob('*.npz'))
  if s['unique_deleted_world_voxels'] and not files:raise ValueError('Deletion archive missing')
  for p in files:
   xyz=np.load(p)['xyz'];frame=int(p.stem)
   # IG archives TSDF lattice corners. Score their affected frozen surfaces,
   # not arbitrary volume corners as if they were object surfaces.
   hit=nearest(initial,xyz)[0]<=np.sqrt(3)*VOXEL+1e-6
   rows.append(dict(key='ig_erasure_'+p.stem,frame=frame,points=initial[hit]))
   input_files.append(p)
  count=int(s['unique_deleted_world_voxels'])
 elif method=='ITSubmap_delete':
  p=root/'counterfactual/summary.json';s=json.loads(p.read_text());input_files.append(p)
  manifest=root/'it_manifest.json';input_files.append(manifest)
  fragments={r['key']:r for r in json.loads(manifest.read_text())['fragments']}
  decisions={d['key']:d for d in s['decisions'] if d['decision']=='retire'}
  for key in s['retired']:
   if key not in decisions or fragments[key]['status']!='inactive':raise ValueError('Unconfirmed retirement')
   path=root/'it_fragments'/key/'surface.npz';input_files.append(path)
   rows.append(dict(key=key,frame=int(decisions[key]['end_frame']),points=np.load(path)['points'],
    source_kind=fragments[key]['kind'],global_ids=fragments[key]['global_ids']))
  count=len(rows)
 else:count=0
 return rows,input_files,count

def deletion_quality(domain,regions,final,cut):
 """Fixed reference atoms plus off-reference atoms define consistent set IoUs."""
 ref=domain['points'];labels=domain['labels'];target=domain['target'];ids=domain['ids'];sizes=domain['sizes']
 final,_=downsample(final)
 audits=[];pred_sizes=[];ious=[];rowids=[];seen=set()
 for record in regions:
  source,_=downsample(record['points'])
  missing=nearest(source,final)[0]>TOLERANCE
  gone=source[missing]
  # A physical location gets only one deletion credit across repeated actions.
  unique=[]
  for p in gone:
   k=tuple(np.floor(p/VOXEL).astype(np.int64))
   if k not in seen:seen.add(k);unique.append(p)
  gone=np.asarray(unique,float).reshape(-1,3)
  valid_time=record['frame']>=cut
  row={k:v for k,v in record.items() if k!='points'}
  row.update(source_samples=len(source),absent_final_samples=int(missing.sum()),unique_absent_samples=len(gone),
   after_rescan_boundary=valid_time,matched_GT=None,IoU=0.)
  audits.append(row)
  if not len(gone):continue
  covered=nearest(ref,gone)[0]<=TOLERANCE
  # Unknown or extra source cells are never silently filtered.
  extra=int((nearest(gone,ref)[0]>TOLERANCE).sum())
  pred_size=int(covered.sum())+extra
  if pred_size==0:pred_size=len(gone)
  inter=np.array([(covered&target&(labels==g)).sum() for g in ids],float)
  if not valid_time:inter[:]=0 # no credit for retiring a still-existing reference object
  iou=inter/(pred_size+sizes-inter)
  ious.append(iou);pred_sizes.append(pred_size);rowids.append(len(audits)-1)
  row.update(prediction_support=pred_size,target_intersections={str(int(g)):int(v) for g,v in zip(ids,inter)})
 matrix=np.asarray(ious).reshape(len(pred_sizes),len(ids))
 score=score_deletions(matrix,sizes,pred_sizes)
 for r,c in score['matches'].items():
  audits[rowids[r]].update(matched_GT=int(ids[c]),IoU=float(matrix[r,c]))
 return dict(**score,executed_regions=len(regions),physically_absent_regions=len(pred_sizes),
  correct_matched_regions=len(score['matches']),per_deletion=audits,per_GT_domain=domain['per_GT'])

def run_highchange(out=None, roots=None, seqs=None):
 data=Path('/root/autodl-tmp/3RScan/3RScan_Dynamic_Additional_20261008')
 raw=Path('/root/autodl-tmp/3RScan/additional_raw_20261008')
 out=Path(out or '/root/autodl-tmp/output/gap_v2_20261010');out.mkdir(parents=True,exist_ok=True)
 ap=json.loads((out/'standard_ap.json').read_text())
 if roots is not None:
  roots={k:Path(v) for k,v in roots.items()}
  for seq,methods in ap['sequences'].items():
   for method in methods:
    if method in roots:
     methods[method]['root']=str(roots[method]/seq)
 report=dict(version='G-AP-v2-event-grounded-20261010',formula='G_AP_t = AP_t + (1-AP_t)*D',
  deletion_definition='D = sum(matched target surface size * deletion IoU) / (fixed removed target size + unmatched executed deletion support)',
  protocol=dict(voxel_m=VOXEL,tolerance_m=TOLERANCE,standard_AP='untouched original evaluator; full precision',
    deletion_GT='raw objectId, all annotated categories; standard AP category filtering unchanged',
    initial='shared IG pre-rescan snapshot; only built targets, final GT occupied surfaces excluded',
    executed='IG actual erasure archives or IT inactive fragment plus retire decision; no natural TSDF disappearance credit',
    outcome='archived action surface must be >5cm from final exported geometry; repeated world cells credited once',
    prediction_groups='executed archive regions/IT fragments; one-to-one Hungarian matching',
    timing='deletion before rescan gets no bonus',aggregation='arithmetic mean of per-scene metrics',
    limitation='D is a removed-object cleanup metric; moved/nonrigid objects receive no bonus; surface sampling and fixed shared build domain affect D'),
  evaluator_sha256=digest(Path(__file__)),standard_AP_evaluator=ap['evaluator_sha256'],sequences={})
 sequence_names=list(seqs) if seqs is not None else list(ap['sequences'])
 for seq in sequence_names:
  methods=ap['sequences'][seq]
  bpath=data/seq/'session_boundary.json';b=json.loads(bpath.read_text());cut=b['rescan_start_frame'];removed=set(map(int,b['changed_instances']['removed']))
  refpath=data/'gt_reference'/(seq+'_vh_clean_2.ply');curpath=data/seq/(seq+'_vh_clean_2.ply')
  rawpath=raw/b['reference_scan_id']/'labels.instances.annotated.v2.ply'
  ref=cloud(refpath);cur=cloud(curpath);labels,nyu=raw_reference(rawpath,ref)
  snap=Path(methods['IGSubmap_delete']['root'])/'correction/reference_surfaces.npz'
  initial=np.load(snap)['all_points']
  domain=target_domain(initial,ref,labels,cur,removed)
  filtered=np.loadtxt(data/'gt_seg_reference'/(seq+'.txt'),dtype=int)
  gt_audit=[dict(GT_id=g,raw_vertices=int((labels==g).sum()),filtered_vertices=int(((filtered>0)&(filtered%1000==g)).sum()),
    NYU40=np.unique(nyu[labels==g]).tolist()) for g in sorted(removed)]
  row=dict(GT_audit=gt_audit,shared_inputs={str(p):digest(p) for p in [bpath,refpath,curpath,rawpath,snap]},methods={})
  for method,a in methods.items():
   root=Path(a['root']);regions,files,count=executed_regions(method,root,initial,cut)
   d=deletion_quality(domain,regions,cloud(root/'final.ply'),cut)
   scores={k:combine(a[k],d['score']) for k in ['AP','AP50','AP25']}
   for v in scores.values():assert v['AP']<=v['G_AP']<=1.
   if not regions:assert all(v['AP']==v['G_AP'] for v in scores.values())
   row['methods'][method]=dict(standard_AP=a,scores=scores,deletion=d,executed_count=count,
     deletion_input_sha256={str(p):digest(p) for p in files})
   print(seq,method,'AP',a['AP'],'D',d['score'],'G_AP',scores['AP']['G_AP'],flush=True)
  report['sequences'][seq]=row
  (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
 report['aggregate']={m:{k:float(np.mean([r['methods'][m]['scores'][k]['G_AP'] for r in report['sequences'].values()])) for k in ['AP','AP50','AP25']} for m in methods}
 (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
 return report

if __name__=='__main__':run_highchange()
