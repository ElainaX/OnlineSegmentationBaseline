# IG 只读矛盾检测：机制对齐与3RScan验证

日期：2026-10-08。仅IG；IT未改动。所有实际删除关闭，本轮没有进行AP评估。

代码位于A100 `/root/OnlineSegmentationBaseline_ig_readonly`，分支 `codex/ig-readonly-detection-20261008`，实验核心提交 `4eca09e`。旧IG纠错分支完整保留。

**检测规则已按本机 onlineSeg e6907b8 对齐。** 检测核心为原文件字节副本，三个地图/成员查询方法的函数体也保持不变。适配层读取当前IG权重和实例映射，不使用上一版逐点3视角投票、60%删除门槛或整图冻结机制。

数据条件仍是现有IG准备序列：224×172、TSDF 25 mm，pair01/03/04每2帧，pair02每5帧；没有改为本机原生适配器的640×480和默认采样。这是机制对齐，不是对用户此前本机实验的数值复现。

## 只读检测结果

|场景|采样间隔|有效检测/采样帧|候选|would_remove|取消|仍待确认|检测耗时(s)|主循环(s)|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|01|2|247/267|6|2|1|3|48.91|157.21|
|02|5|281/315|22|7|8|7|97.81|335.28|
|03|2|283/292|15|6|2|7|77.68|216.25|
|04|2|287/293|18|5|4|9|81.45|206.00|

`would_remove` 只表示检测规则达到确认门槛，不表示该候选已被GT证实，更不表示已经删除。候选是冻结的历史实例引用，数量也不能直接当作不同GT物体数。

检测耗时包含逐帧证据、图片和NPZ保存，不含最终诊断点云导出；主循环包含正常建图、分割和检测。共享GPU及不同运行会影响时间，不把这组数值当作严格的独占GPU加速比。

## reference 与 rescan 分开统计

|场景|reference创建候选|reference确认|reference取消|rescan创建候选|rescan确认|rescan取消|
|---|---:|---:|---:|---:|---:|---:|
|01|0|0|0|6|2|1|
|02|4|0|2|18|7|6|
|03|1|0|0|14|6|2|
|04|2|0|2|16|5|2|

四组reference阶段均无确认标记；确认都发生在rescan阶段。但阶段划分不是GT精度评估，也不能由此断言rescan标记全部正确。确认中没有纯RGB来源的候选，有些包含深度与RGB两类累计证据；这也不等同于证明去掉RGB后结果不会变化。

## 不修改地图的验证

- 8项行为测试通过：源码与查询函数哈希、不同可见区域累计、RGB矛盾、支持重置/取消、遮挡不清空、完整旋转与帧间隔、合并后映射、raw lineage去重/冷却。
- CUDA合成消失测试：中途重编号实例仍能确认would_remove；两次step加一次finalize共3次状态不变性检查通过。
- 真实pair01短序列：12次检测和finalize共13次检查通过。校验包括TSDF属性、块索引、体素成员关系、实例图/特征等，raycast临时工作缓存不计作地图状态。
- 四个完整场景：最终几何与上一轮各自IG关闭检测对照的点数一致，排序后坐标最大差均为0。没有对不同运行的全场景实例AP作相同数值承诺。

|场景|最终点数|与IG对照坐标最大差(m)|实际删除体素|
|---|---:|---:|---:|
|01|101162|0.0|0|
|02|250782|0.0|0|
|03|165369|0.0|0|
|04|191229|0.0|0|

## 失败、重试与限制

- pair04首轮在约frame452长时间停滞，最终主动终止；原日志、部分输出和失败状态保留于 `full/pair_04...`。在GPU4以相同算法/参数重试，仅加进程内堆栈记录，完整完成于 `trace_retry/pair_04...`。停滞没有复现，首轮具体根因尚未确定，不能宣称已修复。
- 早期64×64合成输入触发Open3D默认raycast的CUDA配置错误；当前实际224×172数据与修正尺寸后的合成测试能够运行，未改变原raycast参数。
- 首次状态校验未固定CUDA哈希表枚举顺序，存在把顺序变化误判为内容变化的问题；校验工具已按缓冲区索引排序，只改验证方式。
- 精确位于网格节点的合成平面没有被Open3D抽取出参考点；测试改用不与网格对齐的1.013m平面，未修改检测器阈值。
- 原版检测会跳过无历史地图、有效历史渲染不足和全局矛盾过多的帧。跳过不代表没有矛盾。

|场景|跳过原因与次数|
|---|---|
|01|empty_history: 1, insufficient_historical_render: 19|
|02|empty_history: 1, insufficient_historical_render: 26, widespread_conflict: 7|
|03|empty_history: 1, insufficient_historical_render: 6, widespread_conflict: 2|
|04|empty_history: 1, insufficient_historical_render: 4, widespread_conflict: 1|

## 查看诊断

本机 `results` 内保留汇总、配置、逐帧事件、所有候选的NPZ/元数据，以及确认时的六联图。远端保留完整逐帧图片/NPZ和点云。以下每场景给一个确认帧入口，它们是检测证据示例，不是GT正确性证明：

- pair01，候选2，frame288：[六联图](<E:/Code/github/ig_readonly_20261008/results/pair_01_0cac767c_19f1a892/remove_detection/frames/000288.png>)。
- pair02，候选4，frame1085：[六联图](<E:/Code/github/ig_readonly_20261008/results/pair_02_751a55a1_751a557f/remove_detection/frames/001085.png>)。
- pair03，候选1，frame570：[六联图](<E:/Code/github/ig_readonly_20261008/results/pair_03_f62fd5f8_20c9939d/remove_detection/frames/000570.png>)。
- pair04，候选4，frame456：[六联图](<E:/Code/github/ig_readonly_20261008/results/pair_04_752cc595_55551083/remove_detection/frames/000456.png>)。

代码说明：[IG_READONLY_DETECTION.md](<E:/Code/github/ig_readonly_20261008/port/IG_READONLY_DETECTION.md>)。

下一步纠错前，应将冻结参考点及其深度/RGB矛盾区域与3RScan变化标注逐实例核对，区分物体消失、移动、混合实例及误报；当前尚未执行真实删除或IT实验。
