from pathlib import Path
import json
R=Path('/root/autodl-tmp/output/ig_correction_20261008')
rows=json.loads((R/'results.json').read_text());d={r['seq']:r for r in json.loads((R/'deletion_diagnostics.json').read_text())}
lines=['# IG + 矛盾检测/删除：3RScan pair01–04', '',
       '代码基于 IG disjoint；深度收缩、Carving-AABB、IT 独立几何和 SC 确认均关闭。对照和实验使用相同数据、位姿、mask、合并周期和评估器。', '',
       'AP 为原评估器的 class-agnostic AP。表内使用本轮重新运行的 IG 对照，没有混入旧 baseline/IT 结果。', '',
       '| 场景 | IG：AP / AP50 / AP25 | IG+检测删除：AP / AP50 / AP25 | 候选参考实例 | 达到删除门槛 | 实际删除体素 |',
       '|---|---|---|---:|---:|---:|']
def ap(r):return ' / '.join(f'{r[k]:.3f}' for k in ['AP','AP50','AP25']) if r.get('status')=='complete' else r.get('status','未运行')
for seq in sorted({r['seq'] for r in rows}):
    group={r['variant']:r for r in rows if r['seq']==seq};a=group.get('off',{});b=group.get('delete',{});diag=d.get(seq,{})
    lines.append(f"| {seq.split('_')[1]} | {ap(a)} | {ap(b)} | {diag.get('candidate_count','—')} | {diag.get('retirement_count','—')} | {diag.get('erased_voxel_gt',{}).get('points','—')} |")
lines += ['', '**不能把表中的小幅 AP 差异直接解释为纠错收益。对于实际删除为 0 的场景，删除模块没有改变地图；即使 AP 上升，也不能归因于成功纠错。**', '',
          '## 采样与主循环时间', '', '| 场景 | 间隔 | reference / rescan 采样帧 | TSDF | IG 主循环(s) | 检测删除主循环(s) | 检测耗时(s) |', '|---|---:|---|---|---:|---:|---:|']
for seq in sorted(d):
    g={r['variant']:r for r in rows if r['seq']==seq};a=g.get('off',{}).get('metadata',{});b=g.get('delete',{}).get('metadata',{})
    lines.append(f"| {seq.split('_')[1]} | {b.get('mapping_frame_interval')} | {b.get('reference_sampled_frames')} / {b.get('rescan_sampled_frames')} | {b.get('tsdf_voxel_size_mm')} mm | {a.get('main_loop_seconds',0):.2f} | {b.get('main_loop_seconds',0):.2f} | {d[seq]['audit_seconds']:.2f} |")
lines += ['', 'GPU 为共享机器，主循环耗时受其他作业影响；检测计时是模块内测量，但包含调用同步，不能视为独占 GPU 的严格加速比。', '', '## 检测瓶颈', '']
for seq,q in d.items():
    refs=q['references'];maxfree=max((r['max_confirmed_free'] for r in refs),default=0.)
    lines.append(f"- pair{seq.split('_')[1]}：注册 {len(refs)} 个可靠参考实例；最大多视角确认消失覆盖率 {maxfree:.1%}；{q['guarded_frames']}/{q['audited_frames']} 个检测帧被全局质量保护暂停。")
lines += ['', '候选筛选要求当前至少 100 个可检验的固定表面单元；删除还要求整个参考实例至少 60% 的单元得到多视角确认。低分辨率、局部可见、实例碎片化以及大实例混合都可能使这些门槛难以达到。本轮保留这些原门槛，不根据 AP 临时调整。', '',
          '候选仅是疑似变化，RGB 变化也可触发候选，但不能触发自由空间删除。GT 变化标注在事后诊断中使用，参考 GT 最近点距离上限 10 cm；未标注为变化不保证物理上完全静止，未标注点不计入静态真负例。', '', '## 重复性核查', '']
repeat=R/'off_repeat/result.json'
if repeat.exists():
    x=json.loads(repeat.read_text());lines.append('pair01 追加一次关闭纠错的重复运行：'+ap(x)+'。该重复结果独立保存，不用于替换主表中不利结果。')
lines += ['', '## 文件和可复现性', '',
          '- 远端代码：`/root/OnlineSegmentationBaseline_ig_correction`，分支 `codex/ig-correction-20261008`。',
          '- 远端结果：`/root/autodl-tmp/output/ig_correction_20261008`。',
          '- `results.json`：全部命令、配置核对、AP、耗时、状态、警告。',
          '- `deletion_diagnostics.json`：注册实例、候选拒绝原因、GT 对应与删除诊断。',
          '- `per_instance_comparison.json`：每个 GT 的最佳预测 IoU、纯度和召回，不称作“逐实例 AP”。',
          '- `correction/frames.jsonl`：逐帧候选和状态；`erasure_archive`：实际删除前体素属性。',
          '- `IG_CORRECTION_PROTOCOL.md`：算法、门槛、边界条件和实现区别。', '',
          'CPU 5 项单元测试与真实 CUDA 体素删除/快照隔离测试通过；冒烟测试通过后才运行正式实验。第一轮失败冒烟仅用于修复接口，不计入 AP。']
(R/'IG_CORRECTION_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
