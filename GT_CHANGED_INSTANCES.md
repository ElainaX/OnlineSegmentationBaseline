# 官方 GT 变化实例 PLY

所有文件与之前检测结果使用相同世界坐标（米），没有再次移动或 ICP 对齐。
依据 session_boundary 中已核对过的官方 rigid / removed / nonrigid 标注提取，不依据我们检测器的预测。
读取 prepared all-things GT，不限制在 AP 的 ScanNet18 类别内。没有标注或没有对应顶点的实例不能凭空生成。

## 推荐文件

- `pairXX_gt_changes_reference.ply`：旧 GT 全场景。红=消失，橙=刚体移动/旋转，紫=非刚体变化，黄=同时刚体与非刚体，灰=未标注变化或无实例标签。
- `pairXX_gt_changes_rescan.ply`：当前 GT 全场景，按相同变化类型着色。消失物体没有被伪造到当前场景中。
- `pairXX_gt_changed_instances_reference.ply`：仅旧 GT 的变化实例，每个实例一种颜色。
- `pairXX_gt_changed_instances_rescan.ply`：仅当前 GT 的变化实例，同一物体与旧 GT 保持同色。
- `pairXX_gt_changes_old_new.ply`：仅变化物体的新旧叠加，红=旧位置，青=当前位置。此处红色表示时间，不表示只有消失类别。

PLY 字段：gt_instance_id=当前文件对应时刻的原始 GT ID；reference_instance_id=对应旧 GT ID；session=0旧/1新；change_code 位标志 1消失、2刚体、4非刚体，6表示刚体+非刚体。
不同实例的 RGB 和名字、点数在 manifest.json 中。官方标注是实例级变化，不意味着该实例每一个表面点都发生位移；灰色也不证明物体静止。

## 导出数量与缺失

|场景|旧 GT 变化实例：导出/标注|新 GT 变化实例：导出/标注|旧 GT 缺失 ID|新 GT 缺失 ID|
|---|---:|---:|---|---|
|01|19/20|5/5|[6]|[]|
|02|17/17|16/16|[]|[]|
|03|15/16|9/9|[100]|[]|
|04|14/14|9/9|[]|[]|

pair01 缺失 ID6 的原始语义是 wall，prepared all-things/ScanNet18 标签均排除了它；本机服务器未找到该扫描的原始逐顶点分段文件，不能仅凭包围盒生成假 GT。pair03 的 removed ID100 在已有原始实例标注中就缺失。其他变化实例均已导出，pair04 完整。
少数官方 removed ID 在 rescan 的标签中仍有顶点（详见 manifest 的 removed_ids_still_present_in_rescan_labels）。本导出保留官方变化分类，未自行把这些残留当成新增或移动实例；rescan 背景中的灰色不意味着它们已被证实静止。
