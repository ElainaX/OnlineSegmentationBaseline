# 原始 Flat 评分协议

## 唯一公开入口

```bash
python scripts/evaluate_flat.py outputs/full/flat_run1_run2 --method full
```

该入口先核验两份冻结 GT 的 SHA256，检查方法与 `launch.json` 一致，拒绝 smoke/失败输出，再调用保留原样的 `evaluation/flat/evaluate_flat_outputs.py`。结果写入输出旁的 `flat_metrics.json`；文件已存在时拒绝覆盖。原始函数名中 `H` 表示“要求物理删除证据”的评分路径，供 Full/O0/L/T1-any 使用；`baseline` 是无删除路径，供原版 Baseline 与明确关闭 correction 的 F0 使用。

## 预测文件是一个不可拆分的对

- `final.ply`：最终顶点坐标，保持原顶点顺序。
- `ckpt_final.npz`：布尔/0-1 的 `pred_masks[N_vertices,N_instances]` 与有限一维 `pred_score[N_instances]`。

加载时检查维度、数值和二值 mask；不能只用 PLY 算实例 AP，不能混用其他运行的 checkpoint，也不能排序/抽样 PLY 后仍沿用原 mask 行。

## AP：固定 run2 最终状态域

1. 仅保留 `instance_id>=0` 的已可靠标注 GT 点。**known stuff 仍在预测区域大小和误检中计入**；只有 thing 实例建立正 GT。未知点不进入评价。
2. 每个 GT 点在完整预测点云中找最近顶点，距离 `<=0.05 m` 才建立对应。等距时选较小顶点索引。不做 ICP/平移/旋转/缩放。
3. 把 checkpoint mask 投影到 GT 点。无对应点的所有预测 mask 均为 False，但 GT 点仍保留在 recall 分母，不能按预测覆盖范围裁剪 GT。
4. 主评价保留至少 100 个采样点的 GT 实例和投影后至少 100 点的预测区域。当前固定 GT 有 33 个合格 thing 实例、24 个被尺寸门槛排除的 thing 实例。
5. 在同一 GT 点域计算交并比，使用单一 `object` 类别、checkpoint 原置信度、冻结的 `reference_ap_function.py` 匹配和精确率—召回率积分实现。条件为 **IoU 严格大于阈值**。不要替换为自写的“看起来等价”的贪心匹配/PR 包络积分。
6. 主 AP 是 0.50、0.55、…、0.90 共 9 个阈值 AP 的平均。AP50 单列，AP25 使用 0.25 单列；AP25 和 0.95 均不进入主 AP。

结果同时提供 `secondary_all_sizes`（最小点数 1）的辅助分层、未重建 GT 点数、域外预测支持等诊断。主表只使用 `AP.primary`。

## D：实际删除、时间和最终状态共同决定

评价域是 `derived_change_gt_v1/reference_domain.npz` 的冻结旧表面，分为 target、negative、neutral。这里的 D **不是**“变化实例 voxel 被删比例的均值”。

Full 系列每次真实 erasure 保存原 TSDF 节点档案 `correction/actions/<candidate>/<frame>.npz`，并保存 `*.evidence.npz`：被删节点与六邻域的一环 TSDF、权重、索引及实际删除成员。原始评估器先核验 evidence 与实际档案的索引、TSDF、权重和 lattice 位置一致，再用原 `reference_erasure_surface.py` 重建删除前被影响表面。

- 只给最终点云中距离大于 5 cm、确实消失的受影响表面记删除信用；后来重建回来不记成功。
- 受影响表面与固定旧表面的关联半径为 `sqrt(3)×0.025 m`。
- frame <400 的删除为 before，frame >=400 为 after；多个动作按集合并集统计，不重复计数。
- TP：target 上被 after 成功清除的点。
- FP：before 或 after 删除的 negative 点，加上只在 before 删除、after 没有成功清除的 target 点。
- FN：target 中未成为 TP 的点。
- `D = TP/(TP+FP+FN)`。neutral 与域外删除只披露，不算已知正/负标签。

因此过早的大范围删除不一定得到高 D，T1-any 的低 D 与这个时间/最终状态定义相关。不能拿 `final.ply` 中旧表面是否消失，直接替代真实动作证据。**Full/O0/L/T1-any 缺少证据是 unavailable/错误，不是 D=0**。

## GAP

```text
GAP_observed = AP + (1 - AP) * D_observed
```

AP、D 均用 0..1 的小数代入。Baseline/F0 明确不开删除，D=0，GAP=AP。主表展示百分数；JSON 保留小数。该 Flat conservative observed-domain 协议不等同于其他数据集的 GAP-v3。

## 结果搬迁

Full 系列 `correction/summary.json` 目前含实际动作档案的绝对路径。只把 PLY/checkpoint 搬走足够复算 AP，但不足以验证 D。若需要搬迁完整可评分输出，请整体复制全部 `correction/` 和 evidence，在另一台机器使用相同的绝对输出目录，或对档案引用做经核验的路径重定位；本发布验证在原输出目录完成 D，未自动改写历史动作路径。只让旧路径仍然可访问不够：评估器会拒绝越出当前方法输出根目录的档案引用。

## 已保留的分析测试

`test_flat_scoring.py` 检查完美/空预测、重复实例、stuff 误检、严格阈值、等距对应和未重建 GT。`test_flat_change_scoring.py` 检查提前删除、重复动作、known negative、最终重建、空/缺失档案及 evidence 一致性。

这些测试必须通过；但测试通过不能替代完整建图。原始参考 Baseline 输出经本评估器得到精确 `0.34491254003449123`，是核对实现版本的重要证据。
