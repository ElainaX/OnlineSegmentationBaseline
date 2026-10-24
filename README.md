# Flat Full(H)：可运行代码与消融复现

本分支整理了 A100 上实际使用的 Full(H)、原始 OnlineAnySeg Baseline、F0、O0、L 和 T1-any。默认入口采用 **2.5 cm 全局 TSDF、200000 个块、H 确认、整实例删除与共享节点保护、ObsRefine**；多分辨率、AABB carving、depth shrink 均关闭。

数据、预测 mask、模型权重、GT、点云、checkpoint 和历史日志不在 Git 中。必须准备下文指定的外部资产；仅 clone 代码不能直接开始建图。算法运行不读取 GT。

## 先读这几份文档

| 文档 | 解决的问题 |
|---|---|
| [环境](docs/ENVIRONMENT.md) | 已验证的 Python/CUDA/扩展版本，环境检查 |
| [数据准备](docs/DATA.md) | 原始数据、深度/位姿约定、mask/特征要求、权重和 GT |
| [消融设定](docs/ABLATIONS.md) | 每个方法具体开关、确认规则、删除范围和公平比较条件 |
| [评估协议](docs/EVALUATION.md) | AP、AP50、AP25、D、GAP、证据文件及常见错误 |
| [实测记录](docs/VALIDATION.md) | 本次清理后代码的真实运行结果与验证边界 |

## 从现有共享 mask 复现：完整命令顺序

以下是 **Linux Bash** 命令，在本仓库根目录执行。先按环境文档安装/激活已验证环境，把 `/abs/path/...` 替换为自己的外部文件路径。所有方法共用同一份准备数据和 mask。

```bash
python scripts/verify_release.py
python scripts/check_environment.py

# 1. 使用官方解压目录，仅消费 RGB、depth 和 pose；输出目录必须是新的。
python scripts/prepare_flat.py \
  --raw /abs/path/flat_dataset \
  --output data/flat_run1_run2/frames

# 也可以用 --archive /abs/path/flat_dataset.zip 替代 --raw。
# 2. 复制并检查原实验共享的 191 组预测 mask 和 CLIP 特征。
python scripts/prepare_masks.py \
  --source /abs/path/shared_masks/flat_run1_run2 \
  --output data/masks/flat_run1_run2

# 3. 安装精确版本的 FCGF 权重和两份冻结评价域；自动校验 SHA256。
python scripts/prepare_assets.py \
  --fcgf /abs/path/ResUNetBN2C-16feat-3conv.pth \
  --ap-gt /abs/path/derived_gt_v1/run2_domain.npz \
  --change-gt /abs/path/derived_change_gt_v1/reference_domain.npz

# 4. 对全部输入做解码与格式检查。
python scripts/check_inputs.py \
  --frames data/flat_run1_run2/frames \
  --masks data/masks/flat_run1_run2

# 5. CPU 评分协议测试。
python evaluation/flat/test_flat_scoring.py
python evaluation/flat/test_flat_change_scoring.py
python -m unittest discover -s tests -v

# 6. 可选：前 40 帧的环境/路径测试，结果不能进入正式指标表。
python scripts/run_flat.py --method full --gpu 0 \
  --frames data/flat_run1_run2/frames --masks data/masks/flat_run1_run2 \
  --output outputs/full_smoke --smoke-frames 40

# 7. 正式 Full：完整 762 帧。
python scripts/run_flat.py --method full --gpu 0 \
  --frames data/flat_run1_run2/frames --masks data/masks/flat_run1_run2 \
  --output outputs/full

# 8. 严格原始评分。必须同时有 final.ply 和 ckpt_final.npz。
python scripts/evaluate_flat.py outputs/full/flat_run1_run2 --method full
```

重复运行时换一个新的 `--output`，不要复用已经有内容的目录。评分脚本也禁止覆盖既有 `flat_metrics.json`。查看进度用 `tail -f outputs/full/mapping.log`；完成状态写在 `outputs/full/launch.json`。这些脚本不要求仓库位于某个固定绝对路径。

## 运行其他消融

将上述运行和评分命令中的 `full` 同时改为 `baseline`、`f0`、`o0`、`l` 或 `t1-any`，并使用独立输出目录。例如：

```bash
python scripts/run_flat.py --method o0 --gpu 0 \
  --frames data/flat_run1_run2/frames --masks data/masks/flat_run1_run2 \
  --output outputs/o0
python scripts/evaluate_flat.py outputs/o0/flat_run1_run2 --method o0
```

`configs/ablations.json` 是公开入口的唯一方法表。T1-any 自动设置 `CUDA_LAUNCH_BLOCKING=1`，其 immediate deletion 分支在此 A100 环境需要这个运行设置；因此不要用其耗时直接比较其他方法的速度。

## 到底哪个脚本真正运行算法？

```text
scripts/prepare_flat.py        原始 RGB-D/位姿转换，CPU，无建图
scripts/prepare_masks.py       复制/校验共享预测 mask/特征，无推理、无建图
scripts/prepare_assets.py      校验/安装外部权重与冻结 GT，无建图
scripts/check_inputs.py        完整数据格式检查

scripts/run_flat.py            方法选择、路径检查、独立子进程、运行状态/日志
  full/f0/o0/l -> adapters/run_recorded.py
  t1-any      -> adapters/run_recorded_t1.py
                 -> adapters/run_feng_optimized.py
                 -> run_instdel_multires.py       实际逐帧算法主循环
  baseline    -> adapters/run_baseline.py
                 -> baseline/main.py             原始 Baseline 主循环

scripts/evaluate_flat.py       冻结 GT 校验、方法检查、禁止对 smoke 输出评分
  -> evaluation/flat/evaluate_flat_outputs.py     原始评价入口
  -> flat_scoring.py + reference_ap_function.py   AP
  -> flat_change_scoring.py                      D 与 GAP
```

核心文件名中的 `multires` 是历史名称。当前 Full 的公开入口并不启用多分辨率。部分模块作为 ObsRefine 的父类或导入依赖保留，不能仅凭文件名删除。

## 结果与适用范围

指标是项目定义的固定三维评价域上的类无关实例 AP 和保守变化域上的 D/GAP，**不是官方 Flat 3D benchmark，也不是 COCO AP50:95**。同一协议下的 A100 已验证参考值：Baseline AP 34.4913%，Full AP 36.0535%、D 74.2930%、GAP 83.5613%。全部消融见 [设定与结果](docs/ABLATIONS.md)。

分数必须来自当前 `final.ply` 与配套 `ckpt_final.npz`，不能用旧表中的数字替代实际评分。本次清理后的运行验证和输出校验值以 [实测记录](docs/VALIDATION.md) 为准。GPU/软件栈变化可能改变点顺序或数值；相同指标不等于输出逐字节相同。

## 目录与清理范围

保留算法依赖、Baseline 对照、必要 FCGF 源码/许可、适配层、固定配置、数据工具、原始评估器、测试和复现文档。移除了旧数据集入口、历史实验控制脚本、可视化图片、压缩 GT、重复实验报告及权重文件；没有重写历史分支。`provenance/` 记录整理前来源哈希、发布文件清单与实测结果。
