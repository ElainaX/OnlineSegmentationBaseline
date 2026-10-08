"""Official GT change instances, in the existing aligned reference world frame."""
from pathlib import Path
import json, colorsys, hashlib
import numpy as np
import open3d as o3d

DATA=Path('/root/autodl-tmp/3RScan/3RScan_Dynamic_OnlineAnySeg')
OUT=Path('/root/autodl-tmp/output/ig_deletion_20261008/visualization/gt_changes')
OUT.mkdir(exist_ok=True)
COLORS={0:[180,180,180],1:[255,30,40],2:[255,145,0],4:[175,50,255],6:[255,220,0]}

def write(path,points,colors,**fields):
    n=len(points);a=np.empty(n,dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')]+[(k,'<i4') for k in fields])
    for i,k in enumerate(['x','y','z']):a[k]=points[:,i]
    for i,k in enumerate(['red','green','blue']):a[k]=colors[:,i]
    for k,v in fields.items():a[k]=v
    h=['ply','format binary_little_endian 1.0',f'element vertex {n}','property float x','property float y','property float z',
       'property uchar red','property uchar green','property uchar blue']+['property int '+k for k in fields]+['end_header','']
    with path.open('wb') as f:f.write('\n'.join(h).encode());a.tofile(f)
    check=o3d.io.read_point_cloud(str(path));assert len(check.points)==n
    if n:assert check.has_colors() and np.allclose(np.asarray(check.points),points,atol=1e-6)

def xyz(p):return np.asarray(o3d.io.read_point_cloud(str(p)).points)
def rgb_instance(i):return (np.array(colorsys.hsv_to_rgb((i*.61803398875)%1,.8,1))*255).astype(np.uint8)

manifest=[]
for folder in sorted(DATA.glob('pair_0[1-4]_*')):
    seq=folder.name;short='pair'+seq.split('_')[1]
    boundary=json.loads((folder/'session_boundary.json').read_text());change=boundary['changed_instances']
    before={};after={};new_to_old={}
    for i in set(change['removed']):before[int(i)]=before.get(int(i),0)|1
    for r in change['rigid']:
        a,b=int(r['instance_reference']),int(r['instance_rescan'])
        before[a]=before.get(a,0)|2;after[b]=after.get(b,0)|2
        assert b not in new_to_old or new_to_old[b]==a
        new_to_old[b]=a
    for i in set(change['nonrigid']):
        i=int(i);before[i]=before.get(i,0)|4;after[i]=after.get(i,0)|4
        new_to_old.setdefault(i,i)
    namepath=Path('/root/autodl-tmp/output/ig_readonly_20261008/gt_audit/raw_metadata')/boundary['reference_scan_id']/'semseg.v2.json'
    names={int(x['id']):x['label'] for x in json.loads(namepath.read_text())['segGroups']} if namepath.exists() else {}
    row=dict(seq=seq,coordinate_frame='Existing aligned reference world; meters; no new ICP or object transform',
        boundary_sha256=hashlib.sha256((folder/'session_boundary.json').read_bytes()).hexdigest(),
        official_changes=change,objects=[],sessions={})
    oldnew=[];known_counts={}
    for session,(mapping,ply,labelpath) in enumerate([
        (before,DATA/'gt_reference'/seq/(seq+'_vh_clean_2.ply'),DATA/'gt_seg_reference_all_things'/(seq+'.txt')),
        (after,folder/(seq+'_vh_clean_2.ply'),DATA/'gt_seg_all_things'/(seq+'.txt'))]):
        suffix='reference' if session==0 else 'rescan'
        points=xyz(ply);labels=np.loadtxt(labelpath,dtype=np.int64);assert len(labels)==len(points)
        ids=np.where(labels>0,labels%1000,-1)
        codes=np.array([mapping.get(int(i),0) if i>=0 else 0 for i in ids],np.int32)
        matched=codes>0
        common_ids=np.array([new_to_old.get(int(i),int(i)) if session else int(i) for i in ids],np.int32)
        typecolors=np.array([COLORS.get(int(c),[255,0,160]) for c in codes],np.uint8)
        fields=dict(gt_instance_id=ids,reference_instance_id=common_ids,change_code=codes,session=np.full(len(points),session))
        write(OUT/(short+'_gt_changes_'+suffix+'.ply'),points,typecolors,**fields)
        instcolors=np.array([rgb_instance(int(i)) for i in common_ids[matched]],np.uint8).reshape(-1,3)
        write(OUT/(short+'_gt_changed_instances_'+suffix+'.ply'),points[matched],instcolors,
              **{k:v[matched] for k,v in fields.items()})
        oldnew.append((points[matched],{k:v[matched] for k,v in fields.items()}))
        counts={int(i):int((ids==i).sum()) for i in mapping}
        known_counts[session]=counts
        row['sessions'][suffix]=dict(gt_points=len(points),changed_points=int(matched.sum()),
            annotated_change_instances=len(mapping),exported_change_instances=sum(v>0 for v in counts.values()),
            missing_instances=[i for i,n in counts.items() if not n],instance_point_counts=counts,
            source_points=str(ply),source_labels=str(labelpath))
        if session==1:row['removed_ids_still_present_in_rescan_labels']=[i for i in change['removed'] if (ids==i).any()]
    for i in sorted(before):
        rescan_id=next((int(r['instance_rescan']) for r in change['rigid'] if int(r['instance_reference'])==i),i if before[i]&4 else None)
        row['objects'].append(dict(reference_id=i,rescan_id=rescan_id,name=names.get(i,'unknown'),change_code=before[i],
            rgb=rgb_instance(i).tolist(),reference_points=known_counts[0][i],
            rescan_points=known_counts[1].get(rescan_id,0) if rescan_id is not None else 0))
    points=np.concatenate([x[0] for x in oldnew])
    colors=np.concatenate([np.tile([255,40,40] if s==0 else [0,205,240],(len(x[0]),1)) for s,x in enumerate(oldnew)])
    write(OUT/(short+'_gt_changes_old_new.ply'),points,colors,
        **{k:np.concatenate([x[1][k] for x in oldnew]) for k in oldnew[0][1]})
    manifest.append(row)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
lines=['# 官方 GT 变化实例 PLY','','所有文件与之前检测结果使用相同世界坐标（米），没有再次移动或 ICP 对齐。',
       '依据 session_boundary 中已核对过的官方 rigid / removed / nonrigid 标注提取，不依据我们检测器的预测。',
       '读取 prepared all-things GT，不限制在 AP 的 ScanNet18 类别内。没有标注或没有对应顶点的实例不能凭空生成。','',
       '## 推荐文件','',
       '- `pairXX_gt_changes_reference.ply`：旧 GT 全场景。红=消失，橙=刚体移动/旋转，紫=非刚体变化，黄=同时刚体与非刚体，灰=未标注变化或无实例标签。',
       '- `pairXX_gt_changes_rescan.ply`：当前 GT 全场景，按相同变化类型着色。消失物体没有被伪造到当前场景中。',
       '- `pairXX_gt_changed_instances_reference.ply`：仅旧 GT 的变化实例，每个实例一种颜色。',
       '- `pairXX_gt_changed_instances_rescan.ply`：仅当前 GT 的变化实例，同一物体与旧 GT 保持同色。',
       '- `pairXX_gt_changes_old_new.ply`：仅变化物体的新旧叠加，红=旧位置，青=当前位置。此处红色表示时间，不表示只有消失类别。','',
       'PLY 字段：gt_instance_id=当前文件对应时刻的原始 GT ID；reference_instance_id=对应旧 GT ID；session=0旧/1新；change_code 位标志 1消失、2刚体、4非刚体，6表示刚体+非刚体。',
       '不同实例的 RGB 和名字、点数在 manifest.json 中。官方标注是实例级变化，不意味着该实例每一个表面点都发生位移；灰色也不证明物体静止。','',
       '## 导出数量与缺失','', '|场景|旧 GT 变化实例：导出/标注|新 GT 变化实例：导出/标注|旧 GT 缺失 ID|新 GT 缺失 ID|', '|---|---:|---:|---|---|']
for r in manifest:
    a,b=r['sessions']['reference'],r['sessions']['rescan']
    lines.append(f"|{r['seq'].split('_')[1]}|{a['exported_change_instances']}/{a['annotated_change_instances']}|{b['exported_change_instances']}/{b['annotated_change_instances']}|{a['missing_instances']}|{b['missing_instances']}|")
lines += ['', 'pair01 缺失 ID6 的原始语义是 wall，prepared all-things/ScanNet18 标签均排除了它；本机服务器未找到该扫描的原始逐顶点分段文件，不能仅凭包围盒生成假 GT。pair03 的 removed ID100 在已有原始实例标注中就缺失。其他变化实例均已导出，pair04 完整。',
          '少数官方 removed ID 在 rescan 的标签中仍有顶点（详见 manifest 的 removed_ids_still_present_in_rescan_labels）。本导出保留官方变化分类，未自行把这些残留当成新增或移动实例；rescan 背景中的灰色不意味着它们已被证实静止。']
(OUT/'说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps([dict(seq=r['seq'],sessions=r['sessions'],removed_still_present=r['removed_ids_still_present_in_rescan_labels']) for r in manifest],indent=2))
