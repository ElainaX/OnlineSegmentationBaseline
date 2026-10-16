"""Align reference/rescan in one world, preserve raw IDs and final-rescan GT order."""
import json,zipfile,hashlib,shutil,sys
from pathlib import Path

import argparse
parser=argparse.ArgumentParser(description='Prepare one aligned 3RScan reference/rescan pair; Linux symlinks required.')
parser.add_argument('--work-root',type=Path,required=True)
parser.add_argument('--raw-root',type=Path,required=True)
parser.add_argument('--selection',type=Path,required=True)
parser.add_argument('--template',type=Path,default=Path(__file__).with_name('3rscan_template.yaml'))
args=parser.parse_args()
import numpy as np
import cv2
import open3d as o3d
import yaml
ROOT=args.work_root.resolve(); RAW=args.raw_root.resolve(); DATA=ROOT/'data'
ROOT.mkdir(parents=True,exist_ok=True); DATA.mkdir(exist_ok=True)
(ROOT/'configs').mkdir(exist_ok=True)
selection=json.loads(args.selection.read_text())['selected']
assert len(selection)==1, 'Prepare exactly one pair per invocation'
for item in selection:
    assert not (DATA/item['seq']).exists(), 'Output already exists; inspect before resuming'
    sessions=[]
    for scan in [item['reference'],item['rescan']]:
        with zipfile.ZipFile(RAW/scan/'sequence.zip') as z:
            assert z.testzip() is None
            info_name=next(n for n in z.namelist() if n.endswith('_info.txt'))
            info=dict(s.split(' = ',1) for s in z.read(info_name).decode().splitlines() if ' = ' in s)
        sessions.append(dict(scan_id=scan,raw_frames=int(info['m_frames.size']),info=info))
    item['sessions']=sessions
selection_name=str(args.selection.resolve()); group_name='validation_full'
cfg0=yaml.safe_load(args.template.read_text())
rows=[]

def checked(t):
    assert t.shape==(4,4) and np.isfinite(t).all()
    assert np.allclose(t[3],[0,0,0,1],atol=1e-6)
    assert np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=.01)
    assert np.isclose(np.linalg.det(t[:3,:3]),1,atol=.01)
    return t

def read_ply(path):
    with path.open() as f:
        assert f.readline().strip()=='ply' and f.readline().strip()=='format ascii 1.0'
        properties=[];vertex=False;n=None
        for line in f:
            p=line.split()
            if p[:2]==['element','vertex']:n=int(p[2]);vertex=True
            elif p[:1]==['element']:vertex=False
            elif vertex and p[0]=='property':properties.append(p[-1])
            if p[0]=='end_header':break
        a=np.loadtxt(f,max_rows=n)
    return {k:a[:,i] for i,k in enumerate(properties)}

def gt(scan,t,ply,txt):
    v=read_ply(RAW/scan/'labels.instances.annotated.v2.ply')
    xyz=np.stack([v[k] for k in ['x','y','z']],1)@t[:3,:3].T+t[:3,3]
    color=np.stack([v[k] for k in ['red','green','blue']],1)/255
    ids0=v['objectId']
    assert np.isfinite(ids0).all() and np.equal(ids0,np.floor(ids0)).all()
    assert ids0.min()>=0 and ids0.max()<1000
    ids=ids0.astype(np.int64)
    labels=np.where(ids>0,3000+ids,0)
    pc=o3d.geometry.PointCloud();pc.points=o3d.utility.Vector3dVector(xyz);pc.colors=o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud(str(ply),pc,write_ascii=False)
    np.savetxt(txt,labels,fmt='%d')
    counts={int(i):int((labels==i).sum()) for i in np.unique(labels) if i>0}
    return dict(vertices=len(ids),annotated_instances=len(counts),eligible_gt_instances=sum(n>=100 for n in counts.values()),
                annotated_vertices=int((labels>0).sum()),instance_vertex_counts=counts),xyz

for item in selection:
    seq=item['seq'];dest=DATA/seq
    if (dest/'conversion_complete.json').exists():
        rows.append(json.loads((dest/'conversion_complete.json').read_text()));continue
    for folder in ['color','depth','pose','intrinsic']:(dest/folder).mkdir(parents=True,exist_ok=True)
    for folder in ['gt_seg','gt_reference','gt_seg_reference']:(DATA/folder).mkdir(exist_ok=True)
    transforms=[np.eye(4),checked(np.asarray(item['rescan_metadata']['transform']).reshape(4,4).T)]
    sessions=item['sessions'];infos=[s['info'] for s in sessions]
    h,w=int(infos[0]['m_depthHeight']),int(infos[0]['m_depthWidth'])
    target=np.fromstring(infos[0]['m_calibrationDepthIntrinsic'],sep=' ').reshape(4,4)
    yy,xx=np.mgrid[:h,:w];rays=np.stack([xx,yy,np.ones_like(xx)],-1)@np.linalg.inv(target[:3,:3]).T
    np.savetxt(dest/'intrinsic/intrinsic_depth.txt',target)
    np.savetxt(dest/'intrinsic/intrinsic_color.txt',target)
    all_frames=[];valid=[];session_counts=[];fid=0
    for si,session in enumerate(sessions):
        scan=session['scan_id'];info=session['info'];source=RAW/scan/'sequence';source.mkdir(exist_ok=True)
        with zipfile.ZipFile(RAW/scan/'sequence.zip') as z:
            for member in z.namelist():
                p=(source/member).resolve();assert p.is_relative_to(source.resolve())
            z.extractall(source)
        for key in ['m_calibrationColorExtrinsic','m_calibrationDepthExtrinsic']:
            assert np.allclose(np.fromstring(info[key],sep=' ').reshape(4,4),np.eye(4))
        maps=[]
        for key in ['m_calibrationColorIntrinsic','m_calibrationDepthIntrinsic']:
            k=np.fromstring(info[key],sep=' ').reshape(4,4)[:3,:3];uv=rays@k.T
            maps.append(((uv[:,:,0]/uv[:,:,2]).astype(np.float32),(uv[:,:,1]/uv[:,:,2]).astype(np.float32)))
        count=0
        for source_id in range(session['raw_frames']):
            stem=source/f'frame-{source_id:06d}'
            rgb=cv2.imread(str(stem)+'.color.jpg');depth=cv2.imread(str(stem)+'.depth.pgm',-1)
            pose=np.loadtxt(str(stem)+'.pose.txt')
            assert rgb is not None and depth is not None and depth.dtype==np.uint16
            rgb=cv2.remap(rgb,*maps[0],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            depth=cv2.remap(depth,*maps[1],cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
            depth=np.rint(depth.astype(np.float64)*1000/float(info['m_depthShift'])).clip(0,65535).astype(np.uint16)
            finite=bool(np.isfinite(pose).all())
            if finite:pose=checked(transforms[si]@checked(pose))
            ok=finite and bool(((depth>100)&(depth<5000)).any())
            if ok:valid.append(fid)
            assert cv2.imwrite(str(dest/'color'/f'{fid}.jpg'),rgb)
            assert cv2.imwrite(str(dest/'depth'/f'{fid}.png'),depth)
            np.savetxt(dest/'pose'/f'{fid}.txt',pose)
            all_frames.append(dict(output_frame=fid,session='reference' if si==0 else 'rescan',source_frame=source_id,valid_input=ok))
            fid+=1;count+=1
        session_counts.append(count)
    cut=session_counts[0];valid_set=set(valid)
    strides=[]
    for stride in range(10,0,-1):
        chosen=list(range(0,fid,stride));effective=[i for i in chosen if i in valid_set]
        counts=[sum(i<cut for i in effective),sum(i>=cut for i in effective)]
        if min(counts)>=80:strides.append((stride,chosen,effective,counts))
    if not strides:
        chosen=list(range(fid));effective=[i for i in chosen if i in valid_set]
        strides=[(1,chosen,effective,[sum(i<cut for i in effective),sum(i>=cut for i in effective)])]
    assert len(valid)>0,'No valid input frames'
    stride,chosen,effective,counts=strides[0]
    b=dict(sequence_name=seq,reference_scan_id=item['reference'],rescan_scan_id=item['rescan'],
           reference_frame_count=cut,rescan_frame_count=session_counts[1],total_frame_count=fid,rescan_start_frame=cut,
           stream_order=['reference','rescan'],pose_convention='camera_to_world',coordinate_system='reference_scan_world',
           rescan_to_reference=transforms[1].tolist(),depth_size=[w,h],depth_scale=1000,
           common_intrinsic=target.tolist(),intrinsic_policy='Reproject both calibrated RGB-D sessions to the first reference depth K. RGB linear, depth nearest. No ICP or GT-assisted pose correction.',
           changed_instances={k:item['rescan_metadata'].get(k,[]) for k in ['rigid','removed','nonrigid']},
           split=item['split'],frame_index=all_frames)
    (dest/'session_boundary.json').write_text(json.dumps(b,indent=2))
    refstats,refxyz=gt(item['reference'],transforms[0],DATA/'gt_reference'/(seq+'_vh_clean_2.ply'),DATA/'gt_seg_reference'/(seq+'.txt'))
    finalstats,finalxyz=gt(item['rescan'],transforms[1],dest/(seq+'_vh_clean_2.ply'),DATA/'gt_seg'/(seq+'.txt'))
    # Alignment is diagnostic only. Dynamic changes can reduce surface overlap.
    p=o3d.geometry.PointCloud();p.points=o3d.utility.Vector3dVector(refxyz)
    q=o3d.geometry.PointCloud();q.points=o3d.utility.Vector3dVector(finalxyz)
    distances=np.asarray(q.compute_point_cloud_distance(p))
    report=dict(reference=refstats,final_rescan=finalstats,rescan_to_reference=transforms[1].tolist(),
                aligned_rescan_nearest_reference_median_m=float(np.median(distances)),aligned_rescan_within10cm=float((distances<.1).mean()),
                gt_policy='Final rescan only, transformed to reference world. All positive objectId encoded as 3000+objectId; no NYU40 filtering. No moved-object exclusion.')
    (dest/'gt_report.json').write_text(json.dumps(report,indent=2))
    frames=dest/'frames';frames.mkdir(exist_ok=True)
    for name in ['color','depth','pose','intrinsic']:
        if not (frames/name).exists():(frames/name).symlink_to(dest/name,target_is_directory=True)
    if not (frames/'session_boundary.json').exists():(frames/'session_boundary.json').symlink_to(dest/'session_boundary.json')
    cfg=json.loads(json.dumps(cfg0));cfg['cam'].update(img_h=h,img_w=w,h_crop=0,w_crop=0,bound=False)
    cfg['mapping']['keyframe_freq']=stride;cfg['seg']['seg_add_interval']=stride
    config=ROOT/'configs'/(seq+'.yaml');config.write_text(yaml.safe_dump(cfg,sort_keys=False))
    row=dict(group=group_name,seq=seq,frames=str(frames),masks=str(ROOT/'seg_result'/seq),
             config=str(config),gtroot=str(DATA),gtseg=str(DATA/'gt_seg'),raw_frames=fid,stride=stride,image_shape=[h,w],
             selected_frame_ids=chosen,effective_frame_ids=effective,invalid_selected_frame_ids=sorted(set(chosen)-valid_set),
             selected_frames=len(chosen),effective_frames=len(effective),reference_raw_frames=cut,rescan_raw_frames=session_counts[1],
             reference_effective_frames=counts[0],rescan_effective_frames=counts[1],voxel_size_mm=25,
             gt_vertices=finalstats['vertices'],eligible_gt_instances=finalstats['eligible_gt_instances'],
             config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
             gt_labels_sha256=hashlib.sha256((DATA/'gt_seg'/(seq+'.txt')).read_bytes()).hexdigest())
    rows.append(row);(dest/'conversion_complete.json').write_text(json.dumps(row,indent=2))
    print(json.dumps({k:row[k] for k in ['seq','stride','raw_frames','reference_effective_frames','rescan_effective_frames','eligible_gt_instances']}),flush=True)
(ROOT/'prepared_current.json').write_text(json.dumps(dict(rows=rows,source_selection=selection_name,
    methods=['baseline','current'],depth_shrink=False,carving_aabb=False,correction='method-specific; current delete, baseline off',
    it_extraction_weight_threshold=.5),indent=2))

sys.stdout.flush();sys.stderr.flush();__import__('os')._exit(0)
