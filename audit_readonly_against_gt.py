"""Offline-only GT audit of frozen would_remove evidence. Never changes a map."""
from collections import Counter
import json
import hashlib
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/root/autodl-tmp/output/ig_readonly_20261008')
DATA=Path('/root/autodl-tmp/3RScan/3RScan_Dynamic_OnlineAnySeg')
OUT=ROOT/'gt_audit'
OUT.mkdir(exist_ok=True)
audits=[]
annotation_path=Path('/root/autodl-tmp/3RScan/additional_raw_20261008/3RScan.json')
annotation_bytes=annotation_path.read_bytes()
official_annotations=json.loads(annotation_bytes)
protocol=dict(reference='reference GT, already in reference_scan_world; no extra transform',
    rescan='prepared rescan GT, already transformed into reference_scan_world',
    label_source='gt_seg_reference_all_things; class-agnostic 3000+original objectId, zero remains unknown',
    radii_m=[.025,.05,.10],primary_radius_m=.05,
    annotation_file=str(annotation_path),annotation_sha256=hashlib.sha256(annotation_bytes).hexdigest(),
    strong_support_rule='At least 80% of all frozen points AND 80% of cumulative conflict points match change-labelled reference GT within 5 cm; descriptive grouping, not an official metric',
    mixed_rule='At least 50% of cumulative conflict points match change-labelled GT, but strong-support rule fails',
    denominator='ALL points in each selected region, including unmatched and unlabelled points',
    caveats=['A change-labelled object is not necessarily entirely removed.',
             'Not change-labelled is not proof of physical stability.',
             'No nearby rescan GT vertex is not proof of observed free space.',
             'Prepared all-things labels omit structural surfaces; zero is not a static negative.',
             'Frozen candidates may mix old/new surfaces and several GT objects.'])

for run in json.loads((ROOT/'full/diagnostic_analysis.json').read_text()):
    seq=run['seq']; folder=Path(run['artifact_directory'])/'remove_detection'
    boundary=json.loads((DATA/seq/'session_boundary.json').read_text())
    assert boundary['coordinate_system']=='reference_scan_world'
    annotation_group=next(x for x in official_annotations if x['reference']==boundary['reference_scan_id'])
    annotation_scan=next(x for x in annotation_group['scans'] if x['reference']==boundary['rescan_scan_id'])
    annotation_match={k:annotation_scan[k]==boundary['changed_instances'][k] for k in ['rigid','removed','nonrigid']}
    assert all(annotation_match.values()), 'Prepared change annotation differs from official metadata'
    gt_report=json.loads((DATA/seq/'gt_report.json').read_text())
    changed=boundary['changed_instances']
    types={}
    for kind,ids in [('rigid',[x['instance_reference'] for x in changed['rigid']]),
                     ('removed',changed['removed']),('nonrigid',changed['nonrigid'])]:
        for oid in ids: types.setdefault(int(oid),set()).add(kind)
    names={}
    for group in gt_report['changed_instances'].values():
        for obj in group:
            ref=obj['reference'];names[int(ref['object_id'])]=ref['label']
    metadata_file=OUT/'raw_metadata'/boundary['reference_scan_id']/'semseg.v2.json'
    if metadata_file.exists():
        names.update({int(x['objectId']):x['label'] for x in json.loads(metadata_file.read_text())['segGroups']})
    reference=np.asarray(o3d.io.read_point_cloud(str(DATA/'gt_reference'/seq/(seq+'_vh_clean_2.ply'))).points)
    labels=np.loadtxt(DATA/'gt_seg_reference_all_things'/(seq+'.txt'),dtype=np.int64)
    current=np.asarray(o3d.io.read_point_cloud(str(DATA/seq/(seq+'_vh_clean_2.ply'))).points)
    current_labels=np.loadtxt(DATA/'gt_seg_all_things'/(seq+'.txt'),dtype=np.int64)
    assert len(reference)==len(labels) and len(current)==len(current_labels)
    assert np.all((labels==0)|((labels//1000)==3))
    ref_tree=cKDTree(reference); cur_tree=cKDTree(current)
    summary=json.loads((folder/'summary.json').read_text())
    candidates=[]
    for meta in summary['instances']:
        if meta['state']!='would_remove': continue
        cid=meta['candidate_id']; frame=meta['confirmed_frame']
        file=folder/'candidates'/f'{cid:04d}'/f'{frame:06d}_would_remove.npz'
        with np.load(file) as z:
            points=z['reference_points'].copy()
            regions=dict(reference=np.ones(len(points),bool),
                cumulative_conflict=z['contradicted'].copy(),depth_conflict=z['depth_contradicted'].copy(),
                rgb_conflict=z['rgb_contradicted'].copy(),confirmation_frame_conflict=z['evidence']==2)
        dist,idx=ref_tree.query(points); near_labels=labels[idx]; obj=near_labels%1000
        dcur,icur=cur_tree.query(points)
        stats={}
        for radius in protocol['radii_m']:
            key=str(radius);stats[key]={}
            near=dist<=radius;known=near&(near_labels>0)
            ischanged=known&np.isin(obj,list(types))
            for region,select in regions.items():
                n=int(select.sum()); valid=select&known;counts=Counter(obj[valid].tolist())
                objects=[dict(id=int(oid),name=names.get(int(oid)),points=count,
                    fraction_of_region=count/max(1,n),change_types=sorted(types.get(int(oid),set())))
                    for oid,count in counts.most_common()]
                stats[key][region]=dict(points=n,changed=int((select&ischanged).sum()),
                    not_change_annotated=int((valid&~ischanged).sum()),
                    unlabelled=int((select&near&~known).sum()),unmatched=int((select&~near).sum()),
                    changed_fraction=float((select&ischanged).sum()/max(1,n)),
                    known_fraction=float(valid.sum()/max(1,n)),objects=objects,
                    nearby_rescan_surface=int((select&(dcur<=radius)).sum()),
                    changed_and_no_nearby_rescan_surface=int((select&ischanged&(dcur>radius)).sum()))
                counts_check=stats[key][region]
                assert sum(counts_check[k] for k in ['changed','not_change_annotated','unlabelled','unmatched'])==n
        same_object=[]
        for oid in sorted(set(obj[(dist<=.10)&(near_labels>0)].tolist())):
            selected=(dist<=.10)&(near_labels>0)&(obj==oid)&regions['cumulative_conflict']
            if not selected.any(): continue
            target=current[current_labels==3000+oid]
            if len(target):
                dd=cKDTree(target).query(points[selected])[0]
                details=dict(rescan_points=len(target),same_object_within_5cm=int((dd<=.05).sum()),
                    same_object_within_10cm=int((dd<=.10).sum()),median_distance_m=float(np.median(dd)))
            else:details=dict(rescan_points=0)
            old_surface=reference[labels==3000+oid]
            if len(target) and len(old_surface):
                old_to_new=cKDTree(target).query(old_surface)[0]
                new_to_old=cKDTree(old_surface).query(target)[0]
                details.update(reference_gt_points=len(old_surface),
                    reference_to_rescan_gt_median_m=float(np.median(old_to_new)),
                    rescan_to_reference_gt_median_m=float(np.median(new_to_old)),
                    reference_gt_within_5cm=float((old_to_new<=.05).mean()),
                    reference_gt_within_10cm=float((old_to_new<=.10).mean()),
                    rescan_gt_within_10cm=float((new_to_old<=.10).mean()),
                    centroid_difference_m=float(np.linalg.norm(old_surface.mean(0)-target.mean(0))))
            same_object.append(dict(id=int(oid),name=names.get(int(oid)),change_types=sorted(types.get(int(oid),set())),
                conflict_points=int(selected.sum()),**details))
        primary=stats['0.05']['cumulative_conflict']
        full=stats['0.05']['reference']
        if primary['changed_fraction']>=.8 and full['changed_fraction']>=.8:
            judgement='strong_change_annotation_support'
        elif primary['changed_fraction']>=.5:
            judgement='mixed_change_and_other_objects'
        else:
            judgement='no_change_annotation_support'
        # Explanatory overlays only; do not use rescan point absence as free-space GT.
        if judgement!='strong_change_annotation_support':
            fig,axes=plt.subplots(1,3,figsize=(12,4))
            for ax,(u,v) in zip(axes,[(0,1),(0,2),(1,2)]):
                region_objects=primary['objects'][:2]
                ids=[x['id'] for x in region_objects]
                a=reference[np.isin(labels,[3000+i for i in ids])]
                b=current[np.isin(current_labels,[3000+i for i in ids])]
                ax.scatter(a[:,u],a[:,v],s=2,c='#d04b39',label='Reference GT',alpha=.5)
                ax.scatter(b[:,u],b[:,v],s=2,c='#2377c5',label='Rescan GT',alpha=.5)
                q=points[regions['cumulative_conflict']]
                ax.scatter(q[:,u],q[:,v],s=5,c='#111111',label='Conflict points',alpha=.7)
                ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('XYZ'[u]+' (m)');ax.set_ylabel('XYZ'[v]+' (m)')
                ax.grid(alpha=.2)
            axes[0].legend(fontsize=8)
            desc=', '.join(f"GT{x['id']} {x['name']}" for x in primary['objects'][:2])
            fig.suptitle(f"{seq.split('_')[1]} / C{cid}: {desc}\n{judgement}",fontsize=11)
            fig.tight_layout();fig.savefig(OUT/f"pair{seq.split('_')[1]}_C{cid}_geometry.png",dpi=150);plt.close(fig)
        candidates.append(dict(candidate_id=cid,first_frame=meta['first_frame'],confirmed_frame=frame,
            first_session='reference' if meta['first_frame']<boundary['rescan_start_frame'] else 'rescan',
            metadata=meta,stats=stats,same_object_rescan_check=same_object,judgement=judgement,
            snapshot=str(file)))
        print(json.dumps(dict(seq=seq.split('_')[1],id=cid,frame=frame,
            whole_changed=round(stats['0.05']['reference']['changed_fraction'],3),
            conflict_changed=round(primary['changed_fraction'],3),
            conflict_changed10=round(stats['0.1']['cumulative_conflict']['changed_fraction'],3),
            known=round(primary['known_fraction'],3),objects=primary['objects'][:4]),ensure_ascii=False),flush=True)
    audits.append(dict(seq=seq,changed_object_ids={str(k):sorted(v) for k,v in types.items()},
        object_names=names,official_annotation_match=annotation_match,
        semantic_metadata_source='https://campar.in.tum.de/public_datasets/3RScan/Dataset/'+boundary['reference_scan_id']+'/semseg.v2.json',
        gt_report=gt_report,candidates=candidates))
(OUT/'candidate_gt_audit.json').write_text(json.dumps(dict(protocol=protocol,scenes=audits),indent=2,ensure_ascii=False))
