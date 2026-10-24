# 数据与外部资产

## 1. 建图数据和 GT 必须分开

建图仅需要 RGB、轴向深度、camera-to-world 位姿、相机内参、预测 2D 实例 mask、1024 维 CLIP 特征和 FCGF 权重。官方 `*_segmentation.png`、`*_labels.json`、GT PLY 与本项目 `derived_*_gt_v1` 只属于离线评价来源，不能作为预测 mask 输入。

正式复现路径使用原实验冻结的 **191 组 CropFormer mask/CLIP 特征**，在全部方法间复用。重新推理得到的 mask 是新实验输入，不能保证重现参考分数。

## 2. 官方 Flat 原始包

原发布地址：[ETH Flat dataset](http://robotics.ethz.ch/~asl-datasets/2021_Panoptic_Mapping/flat_dataset.zip)。准备原包或已经完整解压的 `flat_dataset/`：

```text
flat_dataset/
  Intrinsics.txt
  run1/                      400 帧，000000..000399
    000000_color.png
    000000_depth.tiff
    000000_pose.txt
    ...
  run2/                      362 帧，000000..000361
    000000_color.png
    000000_depth.tiff
    000000_pose.txt
    ...
```

源 RGB 是 640×480；TIFF 深度是浮点米制**相机 Z 深度**，不是沿视线的欧氏距离。源 4×4 pose 直接表示 camera-to-world，保持原世界坐标，不做首帧归一化、ICP、轴交换或额外配准。内参固定 `fx=fy=320, cx=320, cy=240`。

`scripts/prepare_flat.py` 按 run1、run2 顺序拼接，不预先抽掉任何帧：

- run1 → 全局编号 0..399；run2 → 400..761。
- RGB 保留原始 PNG 字节，使用 loader 要求的 `.jpg` 文件名；OpenCV 按文件内容正确解码，避免 JPEG 再压缩。
- 深度乘 1000 后四舍五入为 uint16 毫米；非有限/非正/超出 uint16 可表示范围设为 0。运行 loader 再按 `0.1 < depth < 5.0 m` 裁剪。
- pose 原样复制；生成 4×4 intrinsic 与 `session_boundary.json`。
- 算法每 4 帧融合一次，选中 0,4,...,760，共 191 帧；Full 检测器仍逐源帧检查，不应把 762 帧预先缩成 191 帧。

输出：

```text
data/flat_run1_run2/frames/
  color/0.jpg ... 761.jpg
  depth/0.png ... 761.png
  pose/0.txt ... 761.txt
  intrinsic/intrinsic_depth.txt
  session_boundary.json
```

`--raw` 要指向 `flat_dataset` 本身，不是 `run1`。`--output` 要包含最终的 `frames` 层。原始包解压需要额外临时空间；推荐使用已经解压的数据以减少重复磁盘占用。

## 3. 共享预测 mask 与特征

外部 mask 目录必须具有：

```text
shared_masks/flat_run1_run2/
  mask/0.png, 4.png, ..., 760.png
  mask_embeddings/0.pt, 4.pt, ..., 760.pt
  mask_color/                 可选，可视化不参与算法
```

mask 是 480×640 整数实例编号，0 为背景，实例编号从 1 开始。每个 `.pt` 是有限浮点二维 Tensor `[N,1024]`，第 `i-1` 行对应 mask 中实例 `i`。无实例帧也必须保存 `[0,1024]` 空 Tensor 和全 0 mask。特征可由 CUDA 保存；校验脚本显式在 CPU 加载，loader 再移动到所选设备。

```bash
# 原地校验，不复制
python scripts/prepare_masks.py --source /abs/path/shared_masks/flat_run1_run2

# 建立干净、可比较的共享输入副本
python scripts/prepare_masks.py --source /abs/path/shared_masks/flat_run1_run2 \
  --output data/masks/flat_run1_run2
```

该脚本只复制必要的 mask 与特征，排除 `.ipynb_checkpoints`、历史可视化等无关文件，并写入每帧 SHA256。不能用只有 RGB 或只有官方 GT 实例图的目录替代这个输入。

## 4. 可选：从 RGB 重新生成预测 mask

本仓库保存了原始 OnlineAnySeg mask/CLIP 推理脚本及 helpers，哈希与 28018 生成共享输入时使用的脚本一致。完整 Detectron2/CropFormer 项目和两个大型权重仍需外部安装。`--cropformer-root` 应包含 `mask2former/`、`configs/`、`demo_cropformer/predictor.py`，使用支持该 Hornet 配置的原项目版本。

```bash
python scripts/generate_masks.py \
  --cropformer-root /abs/path/detectron2/projects/CropFormer \
  --cropformer-weights /abs/path/Mask2Former_hornet_3x_576d0b.pth \
  --clip-weights /abs/path/open_clip_pytorch_model.bin \
  --data-root data --output-root data/new_masks --gpu 0

python scripts/prepare_masks.py --source data/new_masks/flat_run1_run2
```

特征模型是 `ViT-H-14 / laion2b_s32b_b79k`，保持原始默认置信度 0.6、最小 mask 500 像素、分割间隔 4、480×640 输出。包装器仅增加空实例序列化兼容处理和可选 `--limit-frames` 调试选项，不调整模型阈值。`--limit-frames 1` 可做一次真实推理检查，其不完整输出不能用于正式建图。

新环境需要额外的 Detectron2、CropFormer 自定义算子、open_clip、timm 等依赖。这一路径与“使用已有共享 mask”的最小建图环境分开；其实际验证程度见 `VALIDATION.md`。

## 5. FCGF 权重

必须提供 `ResUNetBN2C-16feat-3conv.pth`，SHA256：

```text
5dfc8d30d08f915b61c133a02d4ac11c90de6eb028e73f273e9c8810f9362496
```

注意历史目录中存在两个同名权重，内容不同；不可凭文件名替换。`prepare_assets.py --fcgf ...` 校验后安装到配置规定的 `third_party/FCGF/model/`。Baseline 和 Full 共用同一个文件，不修改已验证配置中的数值。

## 6. 固定评价域

必须从用户提供的 `derived_gt_v1/` 和 `derived_change_gt_v1/` 原始包取得：

| 资产 | SHA256 |
|---|---|
| `derived_gt_v1/run2_domain.npz` | `7a01a25821e8e9e94044efbbcd6584f8c8f4c33a46a53b7e11574f34a96ff8a9` |
| `derived_change_gt_v1/reference_domain.npz` | `ca1aeb0b319f993a0f36afbfc4106ed66383aadb2540819510ba003762aeb1c5` |

```bash
python scripts/prepare_assets.py \
  --ap-gt /abs/path/derived_gt_v1/run2_domain.npz \
  --change-gt /abs/path/derived_change_gt_v1/reference_domain.npz
```

这两份域来自官方数据的独立 GT 准备过程，已经冻结。AP 域含 `xyz / instance_id / is_thing`；变化域含旧表面 `xyz / instance_id / target / neutral / negative` 等标记。AP 域不是变化域，二者不能互换。本发布不提供一个“重新生成近似 GT”并自动当作同一基准的捷径。没有这两份准确资产可以运行建图，但不能声称复现这套评分协议。

代码、数据与运行输出分别放置；Git 忽略数据、GT、模型权重和生成文件。建议数据与结果放在数据盘，保留足够空间存放 RGB-D、mask 特征和 Full 的逐次删除证据。

## 7. 本次 A100 上的实际资产位置

以下是 2026-10-24 验证时现有的数据位置，便于在同一台 A100 接着复现；它们不是下载地址，也不由 Git 保存。先确认这些外部目录仍然存在，再代入 README 的参数。

| 参数/资产 | A100 上的位置 |
|---|---|
| `--raw` | `/root/autodl-tmp/flat_ablation_pro6000_20261022/raw/official_panoptic/clean_extract/flat_dataset` |
| `--archive` | `/root/autodl-tmp/flat_ablation_pro6000_20261022/raw/official_panoptic/flat_dataset.clean.zip` |
| 共享 mask `--source` | `/root/autodl-tmp/flat_ablation_pro6000_20261022/seg_result/flat_run1_run2_official/flat_run1_run2` |
| 已核验 FCGF | `/root/autodl-tmp/flat_full_release_validation_20261024/repo/third_party/FCGF/model/ResUNetBN2C-16feat-3conv.pth` |
| AP GT | `/root/autodl-tmp/flat_full_release_validation_20261024/repo/evaluation/flat/derived_gt_v1/run2_domain.npz` |
| 变化 GT | `/root/autodl-tmp/flat_full_release_validation_20261024/repo/evaluation/flat/derived_change_gt_v1/reference_domain.npz` |

新机器需自行复制这些数据和权重，再运行哈希/格式校验。GT 与共享 mask 没有在本仓库发布公共下载链接。
