import json
from pathlib import Path

ROOT = Path('/root/autodl-tmp/output/ig_local_evidence_20261008')
data = json.loads((ROOT / 'comparison.json').read_text())
lines = ['# IG 局部证据累积：同图只读对照结果', '',
    '旧、新检测器在同一次 IG 建图中读取相同的融合前地图；无删除，不修改分割、合并或 TSDF。GT 只用于离线诊断。', '',
    '主要统计 rescan 阶段确认、最终仍确认的局部区域（旧检测为不可撤回候选）。5 cm 最近邻、超过全部证据点一半对应某 GT 才关联，场景内 GT 去重。分母为有可用标签的官方 removed。', '',
    '|场景|官方 removed / 可核验|旧检测命中|新区域≥20单元|新区域≥100单元|',
    '|---|---:|---:|---:|---:|']
totals = [0, 0, 0]
denom = 0
for r in data['scenes']:
    n = len(r['verifiable_removed']); denom += n
    hits = [len(r['summaries'][v]['removed_hits']) for v in ('original', 'local_ge20', 'local_ge100')]
    totals = [a + b for a, b in zip(totals, hits)]
    lines.append(f"|{r['seq'].split('_')[1]}|{len(r['official_removed'])} / {n}|" + '|'.join(f'{h}/{n} ({h/n:.1%})' for h in hits) + '|')
lines.append('|合计||' + '|'.join(f'{h}/{denom} ({h/denom:.1%})' for h in totals) + '|')
lines += ['', '## 额外报警与静态对照', '',
    '以下“未标注变化”仅指官方变化列表未包含该 GT，可能是误报、重建表面错误或标注缺漏，不作为确定误报率。区域数与实例数不可直接相除作为精度。', '',
    '|场景|旧检测未标注变化 GT|新≥20 未标注变化 GT|新≥100 未标注变化 GT|参考段旧确认候选|参考段新区域≥20 / ≥100|',
    '|---|---|---|---|---:|---:|']
for r in data['scenes']:
    s = r['summaries']; static = r['local_static_alerts']
    lines.append(f"|{r['seq'].split('_')[1]}|" + '|'.join(str(s[v]['no_change_annotation_ids']) for v in ('original','local_ge20','local_ge100')) + f"|{r['original_static_candidates']}|{static['regions_ge20']} / {static['regions_ge100']}|")
lines += ['', '## 检测开销', '',
    '同图同时运行，时间为各自检测调用累计墙钟时间；共享 A100 有其他负载，不是受控吞吐基准。新方案的计时不含最终连通区域导出。旧检测先运行，新检测可能受益于已预热的 raycast 缓存；旧检测还输出更多 PNG/NPZ 调试文件，所以不能把这些时间直接解释为独立部署的加速倍数。', '',
    '|场景|处理帧数|旧检测秒|新检测秒|新证据单元数|容量拒绝数|确认 / 撤回次数|',
    '|---|---:|---:|---:|---:|---:|---:|']
for r in data['scenes']:
    t = r['runtime']; b = r['bank']
    lines.append(f"|{r['seq'].split('_')[1]}|{t['sampled_frames']}|{t['original_seconds']:.2f}|{t['local_seconds']:.2f}|{b['cells']}|{b['capacity_rejected_new_cells']}|{b['transitions']['confirmed']} / {b['transitions']['revoked']}|")
lines += ['', '## 被命中的消失 GT 与局部覆盖', '',
    '覆盖率是 reference GT 顶点距关联证据区域不超过 5 cm 的比例，不是完整物体删除授权，也不是 AP。', '',
    '|场景|方法|GT ID|证据点数|GT 顶点覆盖率 @5 cm|', '|---|---|---:|---:|---:|']
for r in data['scenes']:
    for v in ('original','local_ge20','local_ge100'):
        for gid, c in r['summaries'][v]['removed_gt_evidence_coverage'].items():
            lines.append(f"|{r['seq'].split('_')[1]}|{v}|{gid}|{c['evidence_points']}|{c['gt_vertex_coverage_5cm']:.1%}|")
lines += ['', '## 解释限制', '',
    '- 本轮是组合机制测试，包含更早保存局部证据、取消实例入口门槛、局部分数累积、撤回以及不使用旧地图否决当前自由空间；不能声称已单独证明累积机制贡献。',
    '- 新方案输出几何变化区域，旧方案输出实例候选；GT 去重后的召回可以比较，候选数量不能直接比较。',
    '- 原检测的 RGB-only 确认不计入深度消失命中。新方案只使用深度；同一几何位置的身份替换无法仅靠深度确认。',
    '- 此轮未执行删除、未声称 AP 提升。需要先处理额外报警，或验证局部删除的准确性，再启用纠错。',
    '- 首轮参数预先固定为分数 [-3,3]、+2 确认、0 撤回，报告两种区域门槛而不按 GT 自动选择。', '']
(ROOT / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
print('totals', totals, 'denominator', denom)
