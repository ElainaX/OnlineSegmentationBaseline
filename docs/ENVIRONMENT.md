# 环境与安装边界

算法运行目标为 Linux + NVIDIA CUDA。Windows 可阅读/编辑代码，但本分支未验证 Windows 原生 MinkowskiEngine/Open3D CUDA 建图。离线 AP/D/GAP 评分只使用 CPU。

## 已实际通过检查的环境

| 项目 | 版本 |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB |
| Python | 3.9.25 |
| PyTorch | 2.1.0+cu118 |
| PyTorch CUDA runtime | 11.8 |
| Open3D | 0.18.0，CUDA 可用 |
| MinkowskiEngine | 0.5.4，CUDA 构建 |
| PyTorch3D | 0.7.9，已编译扩展 |
| torch-scatter | 2.1.2+pt21cu118 |
| NumPy / SciPy | 1.26.4 / 1.13.1 |
| OpenCV | 4.8.1.78 |

其余直接依赖固定在 `requirements-cpu.txt`。这里的版本来自实际环境，不代表任意版本组合都可直接用 pip 从零装好。尤其不能把另一个 CUDA/PyTorch 版本编译的 MinkowskiEngine、PyTorch3D 或 torch-scatter 二进制复制进来。

## A100 上最直接的复现方式

现有通过验证的环境为 `/root/miniconda3/envs/OASeg`。这只是环境位置示例，代码不硬编码这个路径：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate OASeg
python scripts/check_environment.py
```

如需隔离环境，可以克隆已有环境后检查：

```bash
conda create --name flat-full --clone OASeg
conda activate flat-full
python scripts/check_environment.py
```

验证命令会检查 CPU 包、两套 CUDA 后端、实际 torch-scatter GPU 操作和 Open3D GPU→CPU 传输。看到 `ENVIRONMENT_CHECK_PASSED` 后，再按 README 做 40 帧 smoke；完整建图才是最终确认。

## 新机器安装

先准备 Python 3.9、NVIDIA 驱动、CUDA 11.8 开发工具链及编译器，然后安装与 CUDA 对应的 PyTorch 与扩展。例如基础依赖：

```bash
python -m pip install torch==2.1.0 torchvision==0.16.0 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-cpu.txt
python -m pip install torch-scatter==2.1.2 \
  -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

MinkowskiEngine/PyTorch3D 按对应版本的源代码与当前 PyTorch/CUDA 编译。上游安装说明：[MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine#installation)、[PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)、[torch-scatter](https://github.com/rusty1s/pytorch_scatter#installation)。本次验证使用已有编译环境，没有把“在全新系统上重编译所有扩展”计入已通过的检查。

只需要评分时，不安装这些 GPU 扩展：

```bash
python -m pip install -r requirements-cpu.txt
python scripts/check_environment.py --cpu-only
python evaluation/flat/test_flat_scoring.py
python evaluation/flat/test_flat_change_scoring.py
```

## 可选 mask 推理环境

只有 `scripts/generate_masks.py` 需要完整 Detectron2/CropFormer、其自定义算子、`open_clip`、`timm`、CropFormer 权重和 ViT-H-14 权重。本次单帧真实推理使用 Detectron2 0.6、open-clip-torch 3.3.0、timm 1.0.30。使用原实验缓存 mask 时不需要这套额外推理依赖。CropFormer 的配置/权重必须配套，不要替换为另一个分割模型后仍声称是同一输入实验。

## 保留的兼容修复

- Open3D 哈希索引先复制到拥有内存的 CPU NumPy，再传到 PyTorch GPU，并在该路径同步，避免跨框架缓存生命周期导致崩溃。
- Open3D 0.18 raycast 输出尺寸按 range-map factor 补齐后裁剪，保持原有相机/阈值/采样设置。
- FCGF 单元素索引保持 `[1,C]` 维度，避免 NumPy 与 Torch 混合索引降维。
- 核心算法完成并保存输出后，包装器绕过已知 Open3D 进程销毁崩溃；算法中途异常仍返回失败，不能借此把未完成运行当成功。
- T1-any 使用 `CUDA_LAUNCH_BLOCKING=1`，消融语义保持单点即时确认；这会改变耗时。

这些修复不是可调的性能增益开关。不要用删除兼容修复后崩溃的运行，或改了容量/输入的运行，与当前消融表直接比较。
