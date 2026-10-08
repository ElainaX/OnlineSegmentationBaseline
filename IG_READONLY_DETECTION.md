# IG + onlineSeg 只读矛盾检测

本分支从 `10f66dc` 创建，保留此前 IG 纠错实验，未修改 IT。

- `submap_disjoint/readonly_detection_core.py` 是本机 onlineSeg `e6907b8` 的 `remove_detection.py` 原始字节副本。
- `readonly_queries.py` 中三个查询方法原样摘取自该版本，仅构造函数绑定 IG 存储；不向共享基类添加新方法。
- `ig_readonly.py` 只负责读取当前实例映射和权重、接口适配、输出与可选不变性校验。
- `config/ig_readonly_onlineSeg.yaml` 原样提取本机3RScan检测参数；来源与哈希记录在 `ig_readonly_provenance.json`。
- `run_ig_readonly.py` 使用现有 IG 建图、disjoint 合并与导出流程，融合前调用检测器；没有旧 `IGCorrection` 的冻结整图或删除调用。

恢复的机制包括：全程动态发现候选、冻结候选点与颜色、按当前 raw→merged 映射查询归属、基于实例权重筛选、RGB参与矛盾确认、累计不同视角的矛盾点并集、2次有效矛盾观测与20%累计覆盖、有效支持清空证据、2次支持取消、raw mask去重与30帧冷却。不使用16视角/64步窗口，也不要求同一点3次FREE投票。

`would_remove` 只是诊断标签。检测器不会删除实例、清除体素、影响合并决策或修改正常导出。最终粉色诊断点云表示历史事件位置邻域，不表示最终实例身份，也不表示已经移除。

## 运行

```bash
python run_ig_readonly.py --config CONFIG --dataset FRAMES --instance_dir MASKS \
  --output_dir NEW_OUTPUT --seq_name SCENE --merge_selection disjoint_pairs \
  --detection on --detection-config config/ig_readonly_onlineSeg.yaml
```

默认深度收缩、Carving-AABB关闭。`--detection off` 为同入口的无检测对照。新输出目录必须为空。`--verify-readonly` 会在每次检测前后校验已分配 TSDF 属性、块索引、体素成员、实例图和特征等状态；校验非常昂贵，仅用于接口验证，不用于性能比较。CUDA哈希表枚举顺序不稳定，比较前按实际缓冲区索引排序；raycast工作缓存不属于地图内容。

## 对齐范围

检测核心、查询方法和检测参数与记录的本机版本一致，但IG仍使用既有准备数据及采样：pair01/03/04每2帧，pair02每5帧，224×172，25 mm。检测interval=1受建图帧条件约束；观测间隔5仍表示原始帧号差。本分支没有改成原生onlineSeg的640×480输入或其建图/分割采样。元数据明确保留这些差异。

输入地图与实例归属不同，不能保证与本机另一次运行产生完全相同的候选，也不以AP判断检测等价。可复用检测核心供以后IT适配，但本轮仅IG。

## 验证与输出

8项行为测试覆盖原始源码/查询函数哈希、不同可见部分累计、RGB矛盾、有效支持取消、遮挡不清空、完整旋转和帧间隔、合并后归属映射、raw lineage冷却。

真实CUDA合成平面测试验证 `would_remove`、中途实例行号改变及TSDF/语义状态不变；真实pair01短序列另开状态校验。早期64×64测试触发远端Open3D默认raycast的无效CUDA配置，另一次精确落在网格节点的平面无法抽出参考点；失败记录保留。最终测试采用实际3RScan尺寸224×172和不落在网格节点的表面，未更改原检测器或raycast参数。

输出包括 `remove_detection/summary.json`、`events.jsonl`、逐帧六联图与NPZ、候选冻结点和证据、`final_remove_detection.ply`、`remove_reference_instances.ply`、`readonly_alignment.json`。普通最终点云和实例结果仍由原IG导出。

完整pair01–04只读检查由 `run_ig_readonly_suite.py` 调度，不运行AP或删除。初始raycast range-map容量扩展提示单独统计，运行异常、跳过与候选拒绝原因必须保留；不能把跳过等同于无矛盾。
