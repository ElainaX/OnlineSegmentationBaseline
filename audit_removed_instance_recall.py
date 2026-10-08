"""Offline unique-GT-instance recall for official removed annotations.

Confirmed candidate -> majority identity of its accumulated depth-conflict points.
Nearest reference GT among ALL vertices, including unknown/unlabelled vertices.
No parameters of online detection, deletion or AP are changed.
"""
from pathlib import Path
from collections import Counter
import json, hashlib
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

ROOT=Path('/root/autodl-tmp/output/ig_deletion_20261008')
DATA=Path('/root/autodl-tmp/3RScan/3RScan_Dynamic_OnlineAnySeg')
OUT=ROOT/'removed_instance_recall';OUT.mkdir(exist_ok=True)
selected=json.loads((ROOT/'selected_results.json').read_text())
results=[]

for run in sorted(selected,key=lambda r:(r['seq'],r['variant'])):
    if run['variant']!='on':continue
    seq=run['seq'];cmd=run['command'];folder=Path(cmd[cmd.index('--output_dir')+1])/seq
    boundary=json.loads((DATA/seq/'session_boundary.json').read_text())
    changed=boundary['changed_instances']
    removed=set(map(int,changed['removed']))
    rigid={int(r['instance_reference']) for r in changed['rigid']}
    nonrigid=set(map(int,changed['nonrigid']))
    union=removed|rigid|nonrigid
    gt=np.asarray(o3d.io.read_point_cloud(str(DATA/'gt_reference'/seq/(seq+'_vh_clean_2.ply'))).points)
    labels=np.loadtxt(DATA/'gt_seg_reference_all_things'/(seq+'.txt'),dtype=np.int64)
    assert len(gt)==len(labels)
    tree=cKDTree(gt);present=set((labels[labels>0]%1000).tolist())
    counts=Counter((labels[labels>0]%1000).tolist())
    namepath=Path('/root/autodl-tmp/output/ig_readonly_20261008/gt_audit/raw_metadata')/boundary['reference_scan_id']/'semseg.v2.json'
    names={int(x['id']):x['label'] for x in json.loads(namepath.read_text())['segGroups']}
    summary=json.loads((folder/'remove_detection/summary.json').read_text())
    candidates=[]
    for meta in summary['instances']:
        d=folder/'remove_detection/candidates'/f"{meta['candidate_id']:04d}"
        paths=sorted(d.glob('*.npz'))
        if not paths:continue
        artifact=paths[-1];saved=np.load(artifact)
        p=saved['reference_points'];depth=p[saved['depth_contradicted']]
        entry=dict(candidate_id=meta['candidate_id'],state=meta['state'],first_frame=meta['first_frame'],
            confirmed_frame=meta['confirmed_frame'],reference_points=len(p),depth_points=len(depth),
            source=str(artifact),depth_matching={},reference_matching={})
        for kind,points in [('depth',depth),('reference',p)]:
            if not len(points):
                for radius in [.025,.05,.10]:entry[kind+'_matching'][str(radius)]=dict(total_points=0,dominant_id=None,dominant_fraction=0.,majority_id=None,counts={})
                continue
            dist,idx=tree.query(points)
            for radius in [.025,.05,.10]:
                valid=(dist<=radius)&(labels[idx]>0)
                histogram=Counter((labels[idx[valid]]%1000).tolist())
                best,n=histogram.most_common(1)[0] if histogram else (None,0)
                entry[kind+'_matching'][str(radius)]=dict(total_points=len(points),matched_known_points=int(valid.sum()),
                    dominant_id=best,dominant_fraction=n/len(points),majority_id=best if n/len(points)>.5 else None,
                    counts=dict(histogram))
        candidates.append(entry)
    sensitivities={}
    for radius in [.025,.05,.10]:
        # One majority GT identity per confirmed candidate; repeated fragments
        # of one physical object are counted once at scene level.
        association={}
        for c in candidates:
            gid=c['depth_matching'][str(radius)]['majority_id']
            if c['state']=='would_remove' and gid is not None:association.setdefault(gid,[]).append(c['candidate_id'])
        hit=removed&set(association)
        sensitivities[str(radius)]=dict(removed_hit_ids=sorted(hit),removed_hits=len(hit),
            removed_recall_all_annotations=len(hit)/len(removed) if removed else None,
            removed_recall_verifiable=len(hit)/len(removed&present) if removed&present else None,
            any_changed_hit_ids=sorted(union&set(association)),any_changed_hits=len(union&set(association)),
            matched_no_change_annotation=sorted(set(association)-union),gt_to_candidates=association)
    detail=[]
    for gid in sorted(removed):
        associations={state:[] for state in ['would_remove','suspected','cancelled']}
        weak=[]
        for c in candidates:
            # Whole frozen instance identity helps see proposals that did not
            # become confirmed disappearances; never count this as a detection.
            if c['reference_matching']['0.05']['majority_id']==gid:
                associations[c['state']].append(c['candidate_id'])
            n=c['depth_matching']['0.05']['counts'].get(gid,0)
            if n and c['state']=='would_remove':weak.append(dict(candidate=c['candidate_id'],matched_depth_points=n,total_depth_points=c['depth_points']))
        hit=gid in sensitivities['0.05']['removed_hit_ids']
        reason=('hit' if hit else 'GT_label_unavailable' if gid not in present else
            'pending_candidate' if associations['suspected'] else 'cancelled_candidate' if associations['cancelled'] else
            'confirmed_but_no_majority_depth_support' if associations['would_remove'] or weak else
            'no_majority_associated_candidate')
        detail.append(dict(gt_id=gid,name=names.get(gid,'unknown'),reference_gt_vertices=counts.get(gid,0),
            hit=hit,confirmed_hit_candidate_ids=sensitivities['0.05']['gt_to_candidates'].get(gid,[]) if hit else [],
            reference_majority_candidates=associations,confirmed_depth_overlap=weak,stage=reason))
    results.append(dict(seq=seq,source=str(folder),official_removed_ids=sorted(removed),
        official_removed_count=len(removed),verifiable_removed_count=len(removed&present),
        unavailable_removed_ids=sorted(removed-present),any_changed_count=len(union),
        verifiable_any_changed_count=len(union&present),confirmed_candidates=summary['would_remove'],
        sensitivity=sensitivities,removed_details=detail,candidates=candidates))

totals={}
for radius in [.025,.05,.10]:
    hit=sum(r['sensitivity'][str(radius)]['removed_hits'] for r in results)
    all_count=sum(r['official_removed_count'] for r in results)
    verifiable=sum(r['verifiable_removed_count'] for r in results)
    totals[str(radius)]=dict(hits=hit,official_removed=all_count,verifiable_removed=verifiable,
        recall_all_annotations=hit/all_count,recall_verifiable=hit/verifiable)
output=dict(metric='unique GT removed-instance hit recall, custom diagnostic, not official AP',
    primary_radius_m=.05,majority_fraction_strictly_greater_than=.5,
    definition='Confirmed would_remove candidate; >50% of ALL cumulative depth-conflict reference points map within radius to one GT identity; unknown/unmatched remain in denominator. Deduplicate GT IDs within each scene.',
    missing_policy='Report unavailable GT identities separately; all-annotation recall is a conservative count, not evidence that missing-label objects were missed.',
    visibility_policy='GT labels present does not prove sufficient original RGB-D visibility. No visibility-conditioned recall claimed.',
    totals=totals,scenes=results)
(OUT/'removed_instance_recall.json').write_text(json.dumps(output,indent=2,ensure_ascii=False))
lines=['# 官方消失实例：实例级检测召回诊断','','本指标只计算官方 removed，不把移动/非刚体变化也叫作整物体消失。',
    '使用本轮删除开启后的真实在线检测日志，不重新运行建图，不修改任何阈值。','',
    '## 命中规则','',
    '候选必须已经确认 would_remove。取其累计深度矛盾参考点，查找 reference GT 全体顶点中的最近邻；5 cm 内属于同一 GT 实例的点必须超过全部深度矛盾点的 50%。未知/未匹配点保留在分母。RGB-only 候选不算深度消失命中。',
    '多个候选命中同一 GT 只计一次。这是自定义实例级诊断，不是官方 AP，也不表示该 GT 的全部表面都被正确检测或删除。','',
    '|场景|官方 removed|有可用 GT 标签|命中 GT 数|召回率（可核验）|命中 GT ID|',
    '|---|---:|---:|---:|---:|---|']
for r in results:
    s=r['sensitivity']['0.05']
    lines.append(f"|{r['seq'].split('_')[1]}|{r['official_removed_count']}|{r['verifiable_removed_count']}|{s['removed_hits']}|{s['removed_recall_verifiable']:.1%}|{s['removed_hit_ids']}|")
t=totals['0.05'];lines.append(f"|合计|{t['official_removed']}|{t['verifiable_removed']}|{t['hits']}|{t['recall_verifiable']:.1%}|按场景内 GT 去重|")
lines += ['',f"按全部官方标注计数是 {t['hits']}/{t['official_removed']} = {t['recall_all_annotations']:.1%}；不可核验对象不应直接称为确定漏检。",'',
    'pair01 的 wall ID6 被 prepared all-things 标签排除；pair03 ID100 缺少可用原始实例标注。',
    '有标签不意味着扫描中可见或观测充分。因此未命中不能全部归因于矛盾像素判断错误；可能是实例未形成、质量门槛、候选未确认、区域混合或实际观测不足。','',
    '## 半径敏感性','', '|最近邻门槛|命中数/可核验消失 GT|召回率|','|---|---:|---:|']
for radius,t in totals.items():lines.append(f"|{float(radius)*100:g} cm|{t['hits']}/{t['verifiable_removed']}|{t['recall_verifiable']:.1%}|")
lines += ['', '## 各实例在哪一阶段', '', '下表候选归属采用整个冻结参考实例的多数标签；命中判断仍使用上面的累计深度矛盾点规则。无多数关联候选不等价于原始帧完全没有对应 mask。','',
    '|场景|GT ID|名称|GT 顶点数|命中候选|状态诊断|', '|---|---:|---|---:|---|---|']
for r in results:
    for d in r['removed_details']:lines.append(f"|{r['seq'].split('_')[1]}|{d['gt_id']}|{d['name']}|{d['reference_gt_vertices']}|{d['confirmed_hit_candidate_ids']}|{d['stage']}|")
(OUT/'说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(totals=totals,scenes=[{k:r[k] for k in ['seq','official_removed_count','verifiable_removed_count','sensitivity','removed_details']} for r in results]),indent=2,ensure_ascii=False))
