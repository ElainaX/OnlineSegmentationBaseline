from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial import cKDTree

root=Path('/root/autodl-tmp/output/independent_repair_20261007')
report=json.loads((root/'summary.json').read_text())
methods=['global_no_repair','global_clear_tight','global_clear_band','global_clear_target_support','global_replay','independent_deactivate']
labels=['No repair','Clear tight box','Clear padded box','Clear target support','Replay history','Independent submap']
fig,axes=plt.subplots(2,3,figsize=(15,8))
for row,caseid in enumerate(['pair_04_752cc595_55551083','pair_01_0cac767c_19f1a892']):
    case=next(x for x in report['cases'] if x['case']==caseid)
    rows=[next(m for m in case['results'] if m['method']==key) for key in methods]
    ys=[[m['stages'][1]['stale_residual_fraction']*100 for m in rows],
        [m['stages'][1]['lost_previously_supported_static_fraction']*100 for m in rows],
        [1000*m['repair_seconds'] for m in rows]]
    titles=['Old-surface residual (%)','Static support lost (%)','Repair operation (ms)']
    for ax,y,title in zip(axes[row],ys,titles):
        bars=ax.bar(np.arange(6),y,color=['#9fa4ac','#e3b26f','#df8769','#c58dba','#79a9d1','#53a88a'])
        ax.set_xticks(np.arange(6));ax.set_xticklabels(labels,rotation=40,ha='right',fontsize=8)
        ax.set_title(caseid[:7]+' / '+title);ax.grid(axis='y',alpha=.2)
        for b,v in zip(bars,y): ax.annotate('%.2f'%v,(b.get_x()+b.get_width()/2,v),xytext=(0,3),textcoords='offset points',ha='center',fontsize=8)
        ax.margins(y=.2)
fig.suptitle('Oracle geometry-deletion controls (25 mm): immediate result, no new frames\nTiming excludes the separately reported target-support construction; not an AP test',fontsize=12)
fig.tight_layout(rect=(0,0,1,.92)); fig.savefig(root/'repair_controls.png',dpi=160);plt.close(fig)

case=root/'synthetic_final/near_wall'
probes=np.load(case/'evaluation_probes.npz');static=probes['static'];target=probes['target']
# Render a thin horizontal slice so target/wall separation and collateral holes
# can be inspected without hiding them behind a perspective projection.
fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True)
for ax,method,label in zip(axes.flat,methods,labels):
    pc=np.asarray(o3d.io.read_point_cloud(str(case/(method+'_immediate.ply'))).points)
    slab=np.abs(pc[:,1])<.03
    ax.scatter(pc[slab,0],pc[slab,2],s=8,c='#236ea6',label='Remaining map')
    selected=np.abs(static[:,1])<.02
    q=static[selected];missing=cKDTree(pc).query(q)[0]>.0375
    ax.scatter(q[missing,0],q[missing,2],s=12,c='#ee6655',marker='x',label='Missing static surface')
    q=target[np.abs(target[:,1])<.02]
    ax.scatter(q[:,0],q[:,2],s=3,c='#aaaaaa',alpha=.4,label='Old target probes')
    ax.set_title(label);ax.set_xlim(-.6,.6);ax.set_ylim(1.95,2.27);ax.set_xlabel('World X (m)');ax.set_ylabel('World Z (m)');ax.grid(alpha=.15)
axes.flat[0].legend(fontsize=7,loc='lower left')
fig.suptitle('Synthetic near-wall deletion: thin X-Z slice; gray target has disappeared')
fig.tight_layout(rect=(0,0,1,.95));fig.savefig(root/'near_wall_slice.png',dpi=160)
