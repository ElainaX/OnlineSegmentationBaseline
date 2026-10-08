import json
from pathlib import Path
ROOT = Path('/root/autodl-tmp/output/ig_fixed_candidates_20261008')
data = json.loads((ROOT / 'comparison.json').read_text())
names = dict(original='原检测', local_clear_only='只改清零范围', common='逐点计分', extended='逐点计分＋继续观察')
lines = ['# 固定候选后，局部记忆是否有帮助', '',
    '这次不让新算法自己找更多物体。先由原算法给出完全相同的候选名单，再比较它们如何记住同一批证据。地图、候选保存的点、深度/RGB判断和筛选门槛都保持原样；不删除、不改分割。', '',
    '主对照“只改清零范围”：原算法遇到一侧正常，可能清空整个候选记录；这组只清掉当前看到正常的那部分点。确认次数、取消条件、证据求并集方式、20% 覆盖门槛和全部其他规则不变。只用原算法存活期间看到的同一批帧。', '',
    '附加对照“逐点计分”：对同一个位置加减分，分数限制在 -3 到 3，达到 2 才算重复确认。它也只用原算法接受的那些帧，但要求比主对照更严格，因此单列。', '',
    '最后一列再允许原算法确认或取消后继续观察，不把多看几帧的收益混进前两组。', '',
    '## 消失物体检出', '',
    '与前轮相同：只统计官方 removed；深度矛盾点超过一半在 5 cm 内对应某 GT 才算命中，同一个 GT 只计一次。命中是找到局部证据，不代表整个物体都可删除。', '',
    '|场景|可核验消失 GT|原检测|只改清零范围（主对照）|逐点计分|逐点计分＋继续观察|',
    '|---|---:|---:|---:|---:|---:|']
totals = [0,0,0,0]; denominator = 0
for r in data['scenes']:
    n = len(r['verifiable_removed']); denominator += n
    h = [len(r['metrics'][k]['removed_hits']) for k in names]
    totals = [x+y for x,y in zip(totals,h)]
    lines.append(f"|{r['seq'].split('_')[1]}|{n}|"+'|'.join(str(x) for x in h)+'|')
lines.append(f'|合计|{denominator}|'+'|'.join(f'{x} ({x/denominator:.1%})' for x in totals)+'|')
lines += ['', '## 有没有额外报警', '',
    '下表是命中了没有变化标注的 GT 的数量。它们是需要检查的额外报警，不全都能认定为误报；也可能是旧重建错误或标注缺漏。', '',
    '|场景|原检测|只改清零范围|逐点计分|逐点计分＋继续观察|', '|---|---:|---:|---:|---:|']
for r in data['scenes']:
    lines.append(f"|{r['seq'].split('_')[1]}|"+'|'.join(str(len(r['metrics'][k]['no_change_annotation_ids'])) for k in names)+'|')
lines += ['', '## 已经找到候选，为什么还没有确认', '',
    '下面只列候选主要对应消失 GT、但原算法仍未确认的对象。“点太少”是后续某一帧能可靠检查的点不足 100 个；不是说这个物体总共不足 100 个点。这些帧会被跳过，无法凑成第二次有效确认。', '',
    '|场景|GT ID|候选 ID|已接受的矛盾观测次数|后续点太少而跳过|视角或间隔太近而跳过|',
    '|---|---:|---:|---:|---:|---:|']
for r in data['scenes']:
    for p in r['pending_diagnostics']:
        reasons=p['reason_counts']
        lines.append(f"|{r['seq'].split('_')[1]}|{p['gt_id']}|{p['candidate_id']}|{p['accepted_contradiction_observations']}|{reasons.get('paused_insufficient_reference_points',0)}|{reasons.get('paused_correlated_observation',0)}|")
lines += ['', '## 每个候选发生了什么', '',
    'GT 表示该候选当前深度证据主要对应的真实物体。未确认时仍展示关联，方便查看被门槛挡住的内容；此时不算检出。', '',
    '|场景|候选|原检测状态 / GT|只改清零范围状态 / GT|逐点计分状态 / GT|继续观察状态 / GT|', '|---|---:|---|---|---|---|']
for r in data['scenes']:
    for rows in zip(*(r['predictions'][k] for k in names)):
        lines.append(f"|{r['seq'].split('_')[1]}|{rows[0]['candidate_id']}|"+'|'.join(f"{x['state']} / {x['gt_id']}" for x in rows)+'|')
lines += ['', '## 对照检查', '',
    '- 原算法负责候选名单和创建时间；新算法不能新增候选、影响合并、修改地图或触发删除。',
    '- 每次原算法更新，都用未经修改的规则再算一遍，逐项比较状态和证据记录；不一致立即报错。',
    '- 主对照向两种规则提供完全相同的帧和证据，按各自判断确认或取消；逐点计分组进一步固定原算法接受的帧号。额外观察在独立一列中报告。',
    '- RGB 与深度的判断方式没有改变。只有 RGB 的报警不算消失 GT 命中。',
    '- 全部逐帧证据已保存，可不重新建图就重放其他记忆规则。', '']
for r in data['scenes']:
    lines.append(f"- {r['seq']}: 候选 {r['candidates']}，原规则复算检查 {r['checks']['original_replay_checks']} 次，帧筛选检查 {r['checks']['common_frame_checks']} 次。")
(ROOT / 'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print('totals',totals,'denominator',denominator)
