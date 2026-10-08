from pathlib import Path
import json
import numpy as np

root=Path('/root/autodl-tmp/output/ig_deletion_20261008')
rows=json.loads((root/'comparison.json').read_text())
assert len(rows)==4 and all('delta_percentage_points' in r for r in rows)
instances=json.loads((root/'per_instance_iou.json').read_text())
audit=json.loads((root/'erased_voxel_gt_audit.json').read_text())
text=['# IG 局部矛盾删除：3RScan pair01–04 实验', '',
'检测核心沿用 `94e4dde` 的 onlineSeg 只读检测；删除实现初版 `cd2bec8`，空集合修复 `c529f1d`。',
'分支：`codex/ig-confirmed-local-deletion-20261008`。原 IG、IT、baseline 结果保留。', '',
'## 本轮 AP 对比', '',
'以下均为完整 rescan GT 的 class-agnostic 评估；没有排除移动实例。每场景重新运行未删除对照。', '',
'| 场景 | 未删除 AP | 删除 AP | 未删除 AP50 | 删除 AP50 | 未删除 AP25 | 删除 AP25 |',
'|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
    a,b=r['variants']['off'],r['variants']['on']
    text.append(f"| {r['seq'].split('_')[1]} | {a['AP']:.3f} | {b['AP']:.3f} | {a['AP50']:.3f} | {b['AP50']:.3f} | {a['AP25']:.3f} | {b['AP25']:.3f} |")
means={mode:{k:float(np.mean([r['variants'][mode][k] for r in rows])) for k in ['AP','AP50','AP25']} for mode in ['off','on']}
a,b=means['off'],means['on']
text.append(f"| 四场景算术平均 | {a['AP']:.4f} | {b['AP']:.4f} | {a['AP50']:.4f} | {b['AP50']:.4f} | {a['AP25']:.4f} | {b['AP25']:.4f} |")
text += ['', '算术平均用于描述四个场景，不是把所有场景联合评估得到的官方 AP。', '',
'本轮平均 AP 下降 0.775 个百分点，AP50 上升 2.75 个百分点，AP25 上升 0.55 个百分点。不能称为整体稳定提升，暂不将删除设为主线默认。', '',
'pair04 的 GT17 最佳 IoU 从 0.9327 降到 0.6163，覆盖率从 94.98% 降到 62.10%，纯度从 98.11% 到 98.79%。它的主要表象是覆盖下降，而不是纯度下降。尚未区分直接误删、后续融合/关联变化和评估器投影的影响。', '',
'## 删除规则', '',
'本轮是确认后的局部删除，不是把整条实例 track 清空。原检测器的 `would_remove` 条件不变：', '',
'1. 只处理冻结候选中累积过深度矛盾的参考点，RGB-only 矛盾不能清几何。',
'2. 仅提出参考表面的八个 TSDF 插值角点，不扩展到整 AABB 或整个截断带。',
'3. 当前有效深度需再次确认这些位置在自由空间；遮挡、孔洞、深度边界均不能通过。',
'4. 体素的所有有效 raw-mask 成员必须属于冻结候选；新 mask 即使合入相同 merged ID，也不能被旧候选授权删除。',
'5. 写出旧 TSDF/weight/color 与归属存档后清零，清理实例坐标、哈希成员、可见性和原始 mask 几何缓存，删除空 track，重算稀疏匹配边并使 FCGF/导出缓存失效。', '',
'实例级确认有多帧要求，但每个角点并不保证独立满足相同的多帧次数；角点要求“历史深度矛盾 + 当前自由空间”。历史语义特征与观测次数保留，不声称可以逆向拆开之前的特征池化。', '',
'## 配置、删除量与耗时', '',
'IG disjoint 开启；Carving-AABB、深度收缩关闭；25 mm。RGB-D 建图和分割使用相同采样间隔。', '',
'| 场景 | 间隔 | 采样帧 | 未删除主循环 s | 删除主循环 s | 删除处理累计 s | 删除帧数 | 不同被清除体素位置 |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    a,b=r['variants']['off'],r['variants']['on'];d=r['deletion']
    text.append(f"| {r['seq'].split('_')[1]} | {b['mapping_interval']} | {b['selected_frames']} | {a['main_loop_seconds']:.2f} | {b['main_loop_seconds']:.2f} | {d['seconds']:.2f} | {d['deletion_frames']} | {d['unique_world_voxels']} |")
text += ['', '主循环包含原有检测。删除处理时间包含候选筛选、归属清理和稀疏边重算。GPU 有其他任务，运行时间不能当作受控性能结论。pair01 重跑使用 GPU1，对照 GPU3；其余同场景两组使用同一 GPU。', '',
'## 逐实例检查', '',
'以下是评估器投影至 GT 顶点后的最佳实例 IoU，非“单实例 AP”。每组独立选择最高 IoU 预测，不能认为预测序号代表跨运行同一实例。', '',
'| 场景 | GT ID | 未删除最佳 IoU | 删除最佳 IoU | 差值 |',
'|---|---:|---:|---:|---:|']
for r in sorted(instances,key=lambda x:abs(x['delta_best_iou']),reverse=True)[:12]:
    text.append(f"| {r['seq'].split('_')[1]} | {r['gt_id']} | {r['off']['iou']:.4f} | {r['on']['iou']:.4f} | {r['delta_best_iou']:+.4f} |")
text += ['', '完整逐实例结果见 `per_instance_iou.json`。原评估器包含 ICP 微调和 GT→预测点的 15 cm 最近邻映射，保持历史协议不变；远离当前 GT 的旧残影可能不进入同等程度的 AP 惩罚。因此 AP 不等价于残影清除率。', '',
'## 删除位置与官方变化标注', '',
'对被清除的不同 TSDF 角点，查找全部 reference GT 的最近邻，距离门槛 5 cm；分母保留未知与未匹配角点。这是删除位置诊断，不是表面精度或检测 precision。', '',
'| 场景 | 已删除位置 | 靠近有变化标注物体 | 靠近未标注变化物体 | 靠近无标签表面 | 超过 5 cm |',
'|---|---:|---:|---:|---:|---:|']
for r in audit:
    text.append(f"| {r['seq'].split('_')[1]} | {r['unique_erased_corners']} | {r.get('near_change_annotated',0)} | {r.get('near_not_change_annotated',0)} | {r.get('near_unlabelled',0)} | {r.get('far',0)} |")
text += ['', '没有变化标注不严格等价于真实静止；但这些删除区域需要复查，不能因总 AP 增加就忽略误删风险。', '',
'## 复现性和失败记录', '',
'原 GPU 建图存在运行波动。首次实际删除之前，两组的 merge 数量轨迹就可能不同，所以这里是单次运行观察，不能把全部差值严格归因于删除，也不能以此宣称稳定提升。', '',
'| 场景 | 首次删除帧 | 两组首次 merge 数量分歧帧 | 历史未删除 AP/AP50/AP25 |',
'|---|---:|---:|---|']
for r in rows:
    t=r['repeatability'];h=r['historical_off']
    text.append(f"| {r['seq'].split('_')[1]} | {t['first_erasure_frame']} | {t['first_merge_count_divergence']} | {h['AP']:.3f}/{h['AP50']:.3f}/{h['AP25']:.3f} |")
text += ['', 'pair01 初次删除运行在第 292 帧遇到空 Open3D hashmap 查询错误，修复为空提案直接跳过，4 个回归/CUDA 测试通过。失败日志、已有删除存档保留；重跑单独保存在 `retry_pair01`。没有调整阈值来改善结果。', '',
'只读检测器自己的 `actual_*_deletions=0` 仍表示该检测组件不修改地图；实际删除计数以 `deletion/summary.json` 为准，不能混淆两个模块的计数。', '',
'远程完整结果：`/root/autodl-tmp/output/ig_deletion_20261008`。',
'代码目录：`/root/OnlineSegmentationBaseline_ig_deletion`。']
(root/'IG_DELETION_RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
(root/'macro_means.json').write_text(json.dumps(means,indent=2))
print(json.dumps(means,indent=2))
