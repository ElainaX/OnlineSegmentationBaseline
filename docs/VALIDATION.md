# 清理后代码的 A100 实测

本次验证在 2026-10-24 的独立目录 `/root/autodl-tmp/flat_full_release_validation_20261024` 进行。使用整理后的仓库副本、已有 OASeg CUDA 环境、重新生成的 RGB-D/pose 输入和核验后的共享 mask。不是直接对旧输出重新贴表。

机器可读记录、输出 SHA256 和日志校验值见 [validation.json](../provenance/validation.json)，本发布源码清单见 [release_code_sha256.json](../provenance/release_code_sha256.json)。

## 真实完整运行结果

| 本次重新建图 | 源帧数 | AP % | AP50 % | AP25 % | D % | GAP % |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 762 | 34.491254 | 45.084996 | 57.649667 | 0 | 34.491254 |
| Full(H) | 762 | 36.053518 | 46.251994 | 52.631579 | 74.293044 | 83.561306 |

两个运行均正常退出，产生配套 `final.ply` / `ckpt_final.npz`，并由公开评分入口计算指标。AP、AP50、AP25、D、GAP 均与整理前的严格参考分数完全一致。Baseline 的新输出有 410858 个顶点、56 个实例；Full 有 390523 个顶点、43 个实例。主 AP 会再按协议过滤预测区域大小，因此评估实例数可能小于 checkpoint 实例数。

建图入口的墙钟耗时约为 Baseline 430.36 秒、Full 1392.77 秒；这是本次机器负载和运行设置下的记录，不作为公平速度基准。新 PLY/checkpoint 的字节哈希见 JSON，不能用旧输出的哈希替代。

## 已执行检查

1. 精确 FCGF 和两个 GT 文件的 SHA256 校验通过；CUDA 环境检查通过。
2. `prepare_flat.py --raw` 完成全部 762 帧；RGB、深度、pose、内参与原实验准备输入逐帧一致。
3. `prepare_flat.py --archive` 同样成功，和 `--raw` 输出的 2288 个文件逐字节一致。
4. 191 组共享 mask/1024 维特征复制校验通过，全部图像解码、深度类型、内参和位姿格式检查通过。
5. 原始 AP 与 D 评分协议测试通过；17 项 H/消融策略测试通过。
6. Full、F0、O0、L、T1-any 各完成前 40 帧 smoke，均产生输出并正常退出。正式 Baseline/Full 则完成全部 762 帧并评分。
7. 不完整 smoke 输出被公开评分入口拒绝，没有产生可误入主表的指标。
8. 可选 CropFormer + CLIP 路径完成一帧真实 GPU 推理，产生 480×640 实例图、7 个实例和有限 `[7,1024]` 特征。
9. Full 的 2129 条 H 观测决策通过审计；21 次确认均走时间确认路径，本次数据没有触发双来源首轮提前确认。此事实不能被解释为该数据上已验证 H 相对 T2 的增益。
10. 全部公开 CLI 帮助入口可用；110 个源码/配置/协议文件的发布清单在 A100 校验通过。

## 输出元数据修正及验证范围

检查发现，旧包装器没有调用已有的策略元数据保存函数。最终发布在退出前自动保存 H 策略/适配器哈希，或 T1-any 策略/动态生成源码/适配器哈希。该修改只补充输出归档，不改建图、确认、删除或评分逻辑。

上述 762 帧 Full 运行开始于这一归档修正之前；完成后显式调用相同保存函数补齐元数据，并审计实际记录的全部观测。随后使用**最终包装器**重新执行 Full 和 T1-any 的 40 帧运行，确认自动归档、生成源码和审计均通过。没有将它们写成额外的 762 帧重跑。

本次未验证从全新操作系统编译所有 GPU 扩展；未重新推理完整 191 帧 mask；F0/O0/L/T1-any 的全序列参考分数来自此前 A100 完整运行，本次只做它们的入口/短流程验证。对新分割输入或新环境应重新建图评分，不能承诺完全相同的数值。

## 在 A100 查找本次输出

验证根目录下：

```text
data/flat_run1_run2/frames/       本次重新准备的输入
masks/flat_run1_run2/             核验后的共享 mask/特征
outputs/baseline/launch.json     完整运行状态与实际命令
outputs/baseline/mapping.log
outputs/baseline/flat_run1_run2/  Baseline 最终文件与 flat_metrics.json
outputs/full/launch.json
outputs/full/mapping.log
outputs/full/flat_run1_run2/      Full 最终文件、correction 证据与 flat_metrics.json
```

这些大型运行资产保留在 A100，不推送到 Git。重复按 README 运行时选择新输出目录；已有评分文件不会被覆盖。
