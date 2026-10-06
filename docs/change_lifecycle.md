# 矛盾检测与可逆子图生命周期（2026-10-07）

## 代码核对与位置

GitHub `https://github.com/A765422804/onlineSeg.git` 的远端 HEAD 和唯一 main 分支，
与 Windows `E:\Code\github\冯博\onlineSeg` 的 HEAD 完全相同：
`78befb0ec2b4a34e6ba44c2e562419374b4d5250`。
树对象为 `262041abb1b874c588e3ac5a7715fdb72ab71c81`，工作区无差异。
检查了远端引用、提交树、主循环和相关 Python 文件，未找到矛盾检测模块。
本机这份原始 baseline 未被修改。

实现位于 25911 服务器 `/root/OnlineSegmentationBaseline_baseline_ablation`，
分支 `contradiction-lifecycle-20261007`，起点 `aeb0c2c`。
旧版本另保存为 `archive/pre-contradiction-lifecycle-aeb0c2c`。
原有未提交的 `prepare_recovery_inputs.py` 保留，不纳入本次提交。

新增 `submap_disjoint/contradiction.py`、`submap_disjoint/change_lifecycle.py`、
`run_change_lifecycle.py`、`test_change_lifecycle.py`。
只对 `independent_tsdf.py` 增加 pending 状态；未改 main.py、MaskGraph 或 AP 计算。
这是可运行的独立几何生命周期入口，不是已经默认接入所有 baseline 实验的开关。

## 之前检测代码究竟做了什么

归档目录 `experiments/contradiction_audit_disabled_20261006/`：

1. `audit_3rscan.py`：在当前 RGB-D 融合之前 raycast 旧 TSDF。
   深度均为相机 Z，单位米。容差 `max(0.035, 0.04*z)`。
   当前深度更远是 free-space 残差，更近是 foreground；输出像素比例、数量等诊断。
   其中 GT 最近邻/变化标签用于归属和统计，不能当成独立的预测实例检测准确率。
2. `temporal_vote.py`：用相机位置和朝向的相似性筛选视角，用窗口票数和衰减分数累计实例证据。
   相邻重复视角仍可能累计票数；GT 派生候选的局限已在旧报告中注明。
3. `audit_scannet_world_guard.py`：将矛盾像素反投影到旧地图的世界坐标表面，
   后续重投影同一个点，检查 raycast 是否仍支持该表面，然后累计矛盾。
   旧版本检测后仍融合所有观测，也没有执行实例状态变更。

旧 raycast 从当前测量深度附近的块选取查询范围。当旧物体消失、观测到了更远的背景时，
这些块可能不含旧近处表面。本次改为查询历史片段已分配的块，避免该依赖；
这是针对潜在漏检路径的修正，不是已量化的真实数据准确率提升。

## 三步流程

### 1. 封存历史表面，融合前审计

调用 `register_reference(key)` 关闭历史时间片段，将状态设为 frozen。
至少需要 3 次历史观测和 20 个有效表面采样单元。它不是严格的 TSDF 质量证明，
真实场景仍需进一步审计位姿、表面稳定性和权重。

该片段不再接收当前观测，稳定地作为后续判断的参考。
按 5cm 世界网格采样，每个片段最多保留 20,000 个固定表面点。
采样排序固定，避免 GPU 点提取顺序改变采样集合。

每个采样点重投影后，要求：在相机有效范围内、深度有效、不是深度突变/孔洞边缘、
本片段 raycast 深度与该点相符，并通过当前已发布地图的前表面遮挡检查。

- `observed ≈ old`：support，支持存在。
- `observed > old + tolerance`：free，支持旧表面位置为空。
- `observed < old - tolerance`：occluded，不证明消失。
- 无效、出视野、边缘、不可靠对应：unknown，不计正负票。

所有判断在当前帧融合前完成。不使用 RGB 相似度、语义类别、GT 标签或 GT 变化列表。
代码继续采用旧检测的 3.5cm/4% 深度容差，但不沿用 GT 候选构造。
invalid pose 直接报错，不用上一帧位姿代替来产生变化证据。

### 2. 新观测隔离在 pending 几何中

上游提供互斥的 `临时片段键 -> 2D mask`，每个键拥有独立 TSDF。
历史键禁止写入；pending 不进入 `store.surface()` 的正式地图。
收集至少 3 帧后固定表面样本，同时停止继续扩展这块 pending 几何，避免一个已验证的小核心
批准后面新增、尚未验证的几何。等待期间保留其原始 mask 和来源。

接受与停用分开决定：新遮挡物成立，不代表后面的旧物体消失。
该控制器不负责解决不同位置物体的身份重识别。上游键也不能简单使用每帧局部 mask 序号。
调用者必须区分临时片段与长期实例身份；身份关联不会融合 TSDF 数值。

### 3. 多视角证据决定保留、接受、停用与恢复

对同一固定表面单元保存最近最多 16 个视角的证据，64 个审计 step 后过期。
同一视角判定为相机距离小于 10cm 且光轴夹角小于 10 度。
重复视角更新旧票，不叠加票；视角锚点固定，避免缓慢移动时始终被误归为同一个视角。
每个单元独立记录最近有效观测时间，不能因为看见别的单元就延长旧票寿命。
遮挡和 unknown 不会被当成反对票。

单元成为确认 free/support 的条件：至少 3 个不同视角同向支持，
同向票在该单元有效票中的占比至少 80%。这是启发式共识分数，不是校准概率；
“不同视角”也不等于统计独立或消除了系统性位姿误差。

片段级分母为全部固定表面采样单元，不是当前帧可见像素数：

- 单帧 free 至少 20 个单元且占片段 5%：stable -> suspect，保留几何。
- 确认 free 至少 20 个且覆盖片段 60%：旧片段 -> retired，退出当前地图查询。
- 确认 support 至少 20 个且覆盖 pending 50%：pending -> accepted，并封存几何。
- pending 达到同等消失证据：rejected，不发布。
- retired 后确认 support 覆盖 60%：恢复为 stable/frozen。
- pending 等待 64 step 仍无结论：unresolved，不发布，明确不代表“物体消失”。

这里 step 是实际调用审计的次数。如果每 10 原始帧调用一次，64 step 对应最多约 640 原始帧。
本轮没有做局部片段自动切分；小面积矛盾只进入 suspect，不会删除整个实例。
没有可靠反证或支持时，suspect 可继续存在，不因“很久没看见”判为消失。

retired/rejected/unresolved 都保留原始几何及观测，不物理释放。
只有 retired 旧参考自动审计恢复；rejected/unresolved 临时片段须显式重新评估。

## 使用与输出

最小验证：

```bash
python -m unittest -v test_change_lifecycle test_independent_tsdf
python run_change_lifecycle.py --demo --output /path/to/fresh/demo_output
```

真实 RGB-D 入口：

```bash
python run_change_lifecycle.py --manifest packets/frames.jsonl --output /path/to/fresh/output
python run_change_lifecycle.py --resume previous_output/checkpoint --manifest next/frames.jsonl --output /path/to/fresh/resume_output
```

JSONL 每行含 `frame`、`packet`、`phase`（reference 或 live）。packet 是 NPZ：
`rgb` uint8 HxWx3、`depth_m` 浮点米、`K` 3x3、`pose` 世界从相机的 4x4 刚体变换；
`mask__片段键` 是同尺寸 bool mask。包路径相对 manifest 所在目录解析。
reference 必须在 live 之前，帧号严格递增，图像已经对齐、校正，K 不允许 skew。

live 的 mask 键只能是当前 pending 或新片段键。accepted/rejected 后后续观测需要新键。
控制器返回 `sealed_proposals` 和 `integrated`，调用者能确认哪些 mask 被接收，哪些片段已封存。
该 NPZ 入口不是已有 ScanNet/3RScan 目录的自动适配器；现有 sparse owner 的生产接线和真实
预测轨迹的片段切分仍需下一步实施与评估，不能将当前实现说成主实验管线已经完成切换。

输出 frames.jsonl、summary.json、live_map.ply 和 checkpoint。
检查点含 TSDF 活跃块、mask、相机和来源、固定采样点、各视角逐单元证据与时间、事件日志；
恢复之后继续相同的状态决策，重复帧会被拒绝。

## 验证和结果边界

21 项单元/集成回归（15 项新测试 + 6 项旧独立 TSDF 回归），含 CUDA 真正融合和 raycast：

- 静态/遮挡/深度缺失不误停用，局部小矛盾不删除整片段。
- 相同视角不能刷票，缓慢移动最终能构成不同视角，逐单元票会独立过期。
- 消失达到阈值后停用，再次出现后恢复。
- 接受新遮挡物不联动删除旧物体；临时错误几何可拒绝，超时可保留 unresolved。
- pending 取样后不再扩展未审计几何。
- 保存恢复以及 JSONL 入口连续运行得到相同判断。
- 非法位姿、冲突 mask、写历史键、容量上限显式失败。
- 旧独立 TSDF 的链接不融合、停用不破坏邻居等回归继续通过。

合成 demo：static 保持 stable，occluded 保持 stable；
disappeared_then_returned 发生 stable -> suspect -> retired -> stable；
new_occluder 中 new accepted，而 old 仍 stable。
这是解析平面上的正确性检查，不是 ScanNet/3RScan 检测准确率、AP 或速度 benchmark。
没有运行完整场景 AP，也没有调整已有 AP 评估。

测试发现：恰好在体素网格零交叉上的理想平面可能被 Open3D 点云提取遗漏，
测试夹具使用偏离网格的 2.013m/1.013m 平面；真实空表面仍显式拒绝注册，未绕过检查。

## 可能提升、可能降低，以及下一步要验证的部分

可能提升：独立旧参考不会被当前错误观测覆盖；当前深度落点不再限制旧表面 raycast；
遮挡与消失明确分开；同一视角不重复增票；停用可恢复；新旧实例的状态不强制绑定。
这些是实现的机制和受控验证，不能代替真实数据上的误报率/漏报率证据。

可能降低：初期临时几何不发布会暂时降低覆盖；60% 全片段覆盖门槛对仅局部可见物体很保守，
大片段可能长期不退役；三个视角会引入延迟；系统性位姿误差仍可产生一致的假矛盾；
没有颜色、法向、语义和配准不确定度保护；没有自动局部拆分或未观测形状推断。

空间：每片段最多 20,000 点、16 个视角，逐单元存 uint8 状态和 int64 时间，
证据约 O(片段数 * 点数 * 视角数)，不再随无限历史增长；另有 TSDF、原始 masks 和日志。
最多保留 128 片段，TSDF 属性预算默认 2GiB，超限报错，不静默删除/压缩历史。
属性预算不含全部 Open3D 哈希开销、mask、日志、临时张量，因此不是进程/GPU 总显存保证；
融合后分配可能超限，此时必须停止，不能把异常当作一次可继续的成功调用。

时间：当前每帧对每个受审计片段 raycast 全部已分配块，包括 retired 片段以支持恢复。
采样与证据在 CPU，TSDF/raycast 在 GPU。证据部分约 O(片段数 * 点数 * 视角数)，
raycast 还依赖图像大小、活跃块和射线步进，不能简化为固定 O(1)。
该版优先验证行为，不能声称比 baseline 更快；后续需要视锥筛选、空间索引、分频审计。

下一步真实数据首先报告静态误停用率、变化表面 free 证据覆盖、接受/停用延迟、
错误恢复率、未决比例、原始地图与生效地图的覆盖/纯度及耗时内存；
GT 只用于事后计算这些指标，不能用来生成候选或决定停用。
