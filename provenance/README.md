# 来源与校验记录

- `source_snapshot.json`：从已运行的 A100 Full 源、原始 Baseline 源和 28018 原始评估器提取时的 SHA256。它是**整理前来源**记录，不应当作整理后所有文件必须逐字节不变的断言。
- `release_code_sha256.json`：本发布的 Python、配置和协议文件 SHA256，可检查代码是否改变。
- `reference_metrics.json`：整理前已完成 A100 消融的严格原始协议分数，仅作为对照。
- `validation.json`：整理后代码、重新准备输入、Full/Baseline 全序列重跑及测试的实际结果。

Baseline 源来自仓库提交 `507dfd5dc8fde286c1faab01c60d3c2e2068fe84`。其原始 `main.py` SHA256 为 `f8a45cd905110f6489eb6c04cb839e261a47deb0b83f9f3ba0ac9ac0c46de313`。

Full 核心继承已在 A100 修复并跑通的快照，保留 Open3D/PyTorch 哈希索引拥有内存复制修复。整理主要添加公开入口、数据校验和文档；Baseline 包装器改成仓库相对位置，H 审计器改成新的配置位置，测试公共路径辅助函数被拆出。当前原始评分的四个关键文件保持 28018 原始实现，不用临时重写版积分或 voxel 覆盖评价替换。

最终运行包装器还补上已有 H/T1 策略元数据保存函数的调用；这项归档修正的验证范围见 `docs/VALIDATION.md`。`.gitattributes` 保持源文件原始字节，避免不同平台自动转换换行后使冻结源码/配置的校验值变化。

运行结果的输出文件哈希属于各自运行；指标相同不代表 PLY/checkpoint 字节相同。仓库只保存小型元数据，不包含预测结果、删除档案、GT 或模型权重。数据/结果实际位置由运行时参数和 `launch.json` 决定。
