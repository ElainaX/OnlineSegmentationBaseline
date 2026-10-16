# 当前系统使用与 AP / GAP / D 计算说明

更新：2026-10-16。本文是本分支的推荐入口，覆盖当前 IGSubmap + Feng + ObsRefine，以及 3RScan 全实例类无关评价。旧根目录实验脚本与报告保留用于追溯，其中可能有旧服务器绝对路径和旧指标定义；不要把它们当成当前批处理入口。

## 1. 版本、范围和目录

当前算法源码来自实验版本 `68e1fd603839a8df8aa42cef7ea569291357ef2a` 的完整源码快照，移植到已发布的 `IGSubmap-InstDel` 提交 `93f6d56776271fc4daf7a91566a31c82c8a2cb62` 上。前者是来源标识，不意味着本分支 Git 历史包含该未发布提交。13 个核心文件的逐字节 SHA 见 [current_source_provenance.json](current_source_provenance.json)。原始 Baseline 固定为 `507dfd5dc8fde286c1faab01c60d3c2e2068fe84`。

发布增加了便携调用、数据准备、评估与测试入口，没有改变核心算法或原 AP 匹配/积分函数。当前默认配置：全局 0.025 m、无局部 TSDF、无 AABB carving、无 depth shrink、disjoint-pair 合并；动态 3RScan 开 Feng delete，静态 ScanNet 关 Feng，ObsRefine 开启。文件名 `run_instdel_multires.py` 是历史命名，默认入口不启用 multires。

| 路径 | 职责 |
|---|---|
| `run_instdel_multires.py` | 当前在线处理与导出主程序 |
| `submap_disjoint/sparse_owner.py`, `owner.py`, `sparse_edges.py` | 共享全局几何上的实例成员、合并和关系管理 |
| `Instance.py`, `MaskGraph.py` | 实例、稳定身份与关联基础 |
| `feng_remove_detection.py`, `feng_detector_config.json` | 融合前的深度/RGB 矛盾与独立观测确认 |
| `feng_ig_adapter.py`, `submap_disjoint/ig_correction.py` | 历史来源解析、整实例退役、独占几何擦除与状态同步 |
| `submap_disjoint/observation_refinement.py` | 根据历史深度/mask 支持细化点的实例归属；不移动 XYZ |
| `tools/current_system/run_current.py` | 推荐的当前系统调用入口 |
| `tools/current_system/prepare_pair.py`, `validation_pairs.json` | 3RScan 对齐与固定 110 对清单 |
| `tools/current_system/run_masks.py` | 共享二维 masks/CLIP 特征，含空实例序列化保护 |
| `tools/current_system/run_baseline.py` | 调用独立原始 Baseline checkout |
| `eval/evaluate_seqs.py` | 原始 AP 的几何匹配与 PR 积分 |
| `tools/current_system/gap_core.py` | 固定删除域、实际擦除证据与 D |
| `tools/current_system/metrics.py` | domain / evaluate / pooled 命令 |
| `tools/current_system/test_metrics.py` | 不启动建图的 CPU 协议测试 |

系统输入是 RGB-D、C2W 位姿和二维 masks/特征。参考扫描与重扫描依次输入，在线检测不读取 GT 删除标签。GT 和 changed_instances 仅在离线评价使用。新位置可能形成新实例；系统不保证恢复物体运动轨迹或跨位置统一身份。

## 2. 环境与权重

完整建图、mask 推理和原始 AP 几何匹配需要 **Linux + NVIDIA CUDA**。数据准备使用 Linux 目录符号链接。CPU 可以运行已有匹配记录的 pooled AP 和 D 的集合运算；构建 domain 还需要 Open3D 读点云。

环境基础安装见根目录 [README.md](../README.md) 的 PyTorch、PyTorch3D、CropFormer/Detectron2 和 MinkowskiEngine/FCGF 步骤。它是原始平台安装说明，不是所有 GPU 的通用锁文件。完整运行需要 PyTorch/torchvision、Open3D 0.18.0、MinkowskiEngine、PyTorch3D、torch_scatter、OpenCLIP、Detectron2/CropFormer，以及 requirements.txt 中依赖；另外安装 PyYAML、Pillow、tqdm。必须让 CUDA、PyTorch 和编译扩展相互匹配。

```bash
# 已激活与 GPU 匹配的环境后，安装评价/准备的 Python 依赖。
python -m pip install -r tools/current_system/requirements-eval.txt
python -c "import torch, open3d, MinkowskiEngine, pytorch3d, torch_scatter; print(torch.__version__, torch.cuda.is_available(), open3d.__version__)"
python tools/current_system/test_metrics.py
```

这份 requirements 不是完整建图环境锁。不要把旧 A100 的 CUDA 11.x / PyTorch 2.1 二进制直接当作 Blackwell 可用环境。PRO6000 的迁移涉及编译扩展，且 mask 边界有少量差异，不能宣称跨硬件逐位等价；同一比较的两法必须共享同一套新 masks。

权重按原 README 放置：CropFormer `models/Mask2Former_hornet_3x_576d0b.pth`、CLIP `models/open_clip_pytorch_model.bin`、FCGF `third_party/FCGF/model/ResUNetBN2C-16feat-3conv.pth`。原始数据、额外模型权重、环境、结果点云和服务器凭据不随本次源码发布新增。下载 3RScan 需使用获得授权的官方数据；本入口不代替数据授权或下载器。

运行包装器使用与实验恢复一致的外部保护：Open3D 0.18 range-map 输出尺寸补齐后裁回原图；FCGF 单样本索引维度保护；擦除前读取所有 incident-edge 证据。原算法文件不被重写。`fast` 归属查询提速与只读计时保持开启；结束前保存结果，再绕过已知 Open3D 析构异常。非成功异常仍返回非零，不能仅凭目录存在认定成功。

## 3. 准备一个 3RScan 配对

以下命令从仓库根目录执行，`WORK` 和 `RAW` 使用绝对路径。每次准备一个配对；已存在的数据目录会拒绝覆盖。

```bash
export WORK=/absolute/path/3rscan_run
export RAW=/absolute/path/3rscan_raw
export SEQ=val_095821f7_095821f9
mkdir -p "$WORK"
python tools/current_system/metrics.py select-pair --seq "$SEQ" --out "$WORK/selection.json"
python tools/current_system/prepare_pair.py \
  --work-root "$WORK" --raw-root "$RAW" --selection "$WORK/selection.json"
```

RAW 下需要 selection 中 reference/rescan 两个完整 UUID 目录，每个包含 `sequence.zip` 和 `labels.instances.annotated.v2.ply`。当前转换器支持官方 ASCII 标注 PLY；格式不符会报错。压缩包含 `_info.txt`、RGB、深度与 pose。程序会在对应 RAW UUID 目录解压 sequence，所以该目录须可写。

转换后：

```text
$WORK/
  configs/<SEQ>.yaml
  prepared_current.json                # 最近一次 prepare 的记录，不是全批 manifest
  data/<SEQ>/{color,depth,pose,intrinsic}/
  data/<SEQ>/frames/                   # 指向上面目录的符号链接
  data/<SEQ>/session_boundary.json
  data/<SEQ>/<SEQ>_vh_clean_2.ply       # 最终 rescan GT，参考扫描世界坐标
  data/gt_seg/<SEQ>.txt                # 与 GT PLY 顶点顺序一一对应
```

位姿为 camera-to-world，坐标单位米，深度图为 uint16 毫米。rescan 元数据 transform 按 `reshape(4,4).T` 解释并左乘 rescan C2W；不使用 ICP 或根据分数修正 GT。两次扫描重投影到 reference 深度相机内参，RGB 线性采样、深度最近邻。

全部帧按 reference→rescan 编号。映射 stride 由转换器从 10 到 1 选择，使每段尽可能有至少 80 个有效采样帧，记录在生成的 YAML/准备记录中。**不要套用 ScanNet 的 stride≥10、≤1000 帧规则。**Feng 按原流中的有效位姿帧检查；建图按 mapping.keyframe_freq 采样，不能把流裁成采样列表后仍声称检测节奏相同。

全实例 GT 编码为 `objectId>0 → 3000+objectId`，0 为 ignore；要求 ID 是 [0,999] 内的整数。3 是评价器适配用的类别载体，不代表只评价某个语义类别。墙、地板等有正实例标注的结构也包含；最小 100 个 GT 顶点的规则在原评价器保留。

## 4. 生成共享 masks

先完成原 README 的 CropFormer 安装，把 `scripts/mask_predict/*` 复制到 `third_party/detectron2/projects/CropFormer/demo_cropformer/`。以下从准备记录取 stride 和尺寸，避免手写错配：

```bash
read STRIDE HEIGHT WIDTH < <(python -c 'import json,sys; r=json.load(open(sys.argv[1]))["rows"][0]; print(r["stride"],*r["image_shape"])' "$WORK/prepared_current.json")
python tools/current_system/run_masks.py \
  --root "$WORK/data" --seq_name "$SEQ" --output_root "$WORK/seg_result" \
  --seg_interval "$STRIDE" --dst_h "$HEIGHT" --dst_w "$WIDTH" \
  --pretrained_path "$PWD/models/open_clip_pytorch_model.bin" \
  --opts MODEL.WEIGHTS "$PWD/models/Mask2Former_hornet_3x_576d0b.pth"
```

输出 `seg_result/<SEQ>/mask/*.png` 和 `mask_embeddings/*.pt` 等。两种方法及所有消融读取同一份文件。包装器可跳过已通过形状检查的完整帧，但换输入、权重、stride 或尺寸时必须使用新目录；它不是输入哈希变化检测器。

## 5. 运行当前系统与 Baseline

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python tools/current_system/run_current.py \
  --config "$WORK/configs/$SEQ.yaml" --dataset "$WORK/data/$SEQ/frames" \
  --instance_dir "$WORK/seg_result/$SEQ" --output_dir "$WORK/results/current" \
  --seq_name "$SEQ" --correction delete --observation-refine on
```

静态数据或 F0 消融改为 `--correction off`；O0 改为 `--observation-refine off`；每个变体指定新的 output_dir。RGB0/T1 可复制 detector JSON 并修改对应配置项，再用 `--detector-config /absolute/path/variant.json` 指定，保留其他阈值和评价域。`--merge-selection baseline_components` 只切换当前系统的合并策略，**不等同于原始 Baseline**。`detect` 只检测不执行删除。更细的计划见 [消融plan.md](消融plan.md)，未实现的消融不要通过旧实验脚本名称猜测。

原始 Baseline 使用独立 checkout，防止误调用当前改过的 Instance/MaskGraph：

```bash
git clone https://github.com/ElainaX/OnlineSegmentationBaseline.git /absolute/path/baseline_original
git -C /absolute/path/baseline_original checkout 507dfd5dc8fde286c1faab01c60d3c2e2068fe84
export OASEG_BASELINE_ROOT=/absolute/path/baseline_original
# Baseline 的 FCGF 权重/依赖也按其 README 准备；保持已跟踪源码干净。
python tools/current_system/run_baseline.py \
  -c "$WORK/configs/$SEQ.yaml" -d "$WORK/data/$SEQ/frames" \
  -i "$WORK/seg_result/$SEQ" -o "$WORK/results/baseline" --seq_name "$SEQ"
```

原始 Baseline 包装器保持 seed=27，保留精确的无有效深度保护和主循环计时。它要求正确提交且已跟踪文件无修改。建议 GPU 单任务，方法依次执行。主结果为每个方法下的 `<SEQ>/final.ply` 和 `ckpt_final.npz`；pred_masks 轴顺序为 `[final点数, 实例数]`。不要打乱点云顶点顺序后复用 checkpoint。

当前方法还保存 `correction/summary.json`、`correction/actions/<candidate_id>/<frame>.npz` 及同目录的 `<frame>.evidence.npz`。历史无候选分目录的记录可位于 `correction/erasure_archive/`。候选子目录防止同一帧多次动作互相覆盖。证据文件记录擦除前正权重 TSDF 节点与保留邻居，供离线重建与擦除节点相接的零交叉表面。仅有最终点云或删除实例 ID **不能追溯出同等 D**。

## 6. 先建立共享删除评价域，再评价两个方法

domain 只依赖 GT、输入观察与固定阈值；同一配对仅构建一次，两个方法共用。下面读取 reference UUID 来定位原始标注。

```bash
REF=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"][0]["reference"])' "$WORK/selection.json")
python tools/current_system/metrics.py domain \
  --seq "$SEQ" --boundary "$WORK/data/$SEQ/session_boundary.json" \
  --reference-ply "$RAW/$REF/labels.instances.annotated.v2.ply" \
  --gt-ply "$WORK/data/$SEQ/${SEQ}_vh_clean_2.ply" \
  --frames "$WORK/data/$SEQ/frames" --out "$WORK/domains/$SEQ"

python tools/current_system/metrics.py evaluate \
  --seq "$SEQ" --method current --result "$WORK/results/current/$SEQ" \
  --gt-ply "$WORK/data/$SEQ/${SEQ}_vh_clean_2.ply" \
  --gt-labels "$WORK/data/gt_seg/$SEQ.txt" --domain "$WORK/domains/$SEQ" \
  --out "$WORK/evaluation/current/$SEQ"

python tools/current_system/metrics.py evaluate \
  --seq "$SEQ" --method baseline --result "$WORK/results/baseline/$SEQ" --no-actions \
  --gt-ply "$WORK/data/$SEQ/${SEQ}_vh_clean_2.ply" \
  --gt-labels "$WORK/data/gt_seg/$SEQ.txt" --domain "$WORK/domains/$SEQ" \
  --out "$WORK/evaluation/baseline/$SEQ"
```

`--no-actions` 是明确的无显式擦除声明，仅供 Baseline/Feng-off 使用，不能用它掩盖 Full 缺失的动作日志。命令拒绝该声明与非空 actions 冲突。日志明确实际擦除数为 0 的动作不会产生 archive；其他动作需要的 archive/evidence 缺失都会报错。复制服务器结果到新目录时，保留 correction 下的完整相对目录结构（尤其 candidate_id），入口按本地目录重新定位，无需修改历史日志中的绝对路径。

evaluate 生成独立 `metrics.json`、`ap_matches.pkl`，不写入原建图输出。metrics 包含 AP/AP50/AP25、D、GAP_v3/GAP50/GAP25、TP/FP/FN、删除 precision/recall、输入和 domain SHA。Pickle 仅加载自己生成并可信的中间结果。

现有 2026-10-14 批次的旧匹配 pickle 结构、旧结果 JSON 与本次封装不同，不能直接混入 pooled。重新调用本入口评价保留输出，或编写经过验证的显式迁移；不要手改 protocol 字符串绕过检查。

## 7. AP 的准确含义

原评价器将重建上的实例 mask 投影到最终 rescan GT 顶点，使用距有效预测点 **0.15 m** 内的最近邻。两种方法用同一 GT；0/void、小区域和忽略比例沿用原始实现。根据实例置信度排序，在各 IoU 阈值上构造 PR 并积分。

- AP：IoU **0.50、0.55、…、0.90** 共 9 个阈值的均值；这里不是 COCO 的 0.50–0.95。
- AP50/AP25：对应单阈值，原实现用严格 `IoU > threshold`，不是 `>=`。
- 全实例类无关：不按预测语义类别匹配，也不按 NYU40 过滤 GT；不是官方 200 类/18 类语义 mAP。
- 某场景没有合格 GT 时 AP 未定义，序列化为 null；有 GT 而无正确匹配时 AP=0，保留该场景。
- 原几何匹配器无法处理 checkpoint 中完全没有预测列的情况。本入口会报明确错误，保留为待处理失败，不能静默丢场景或把未运行成功的评价写成 0。此限制不影响仅“有预测但 AP=0”的正常评价。

历史 ScanNet18 过滤的 3RScan 结果与当前全实例结果属于不同协议，不能平均或拼在一起。根目录旧 `eval_gap_v2.py` 等文件不定义本文的 GAP-v3。

## 8. D 是怎么计算的

先明确计数对象：**2.5 cm 空间网格内的表面原子，不是删除实例数，也不是所有正权重体积节点数。**参考 GT 点用 `floor(x/0.025)` 分组，每组取质心、以最近原始点赋实例 ID。

目标集合 T 满足全部条件：

1. 属于标注 removed 或 rigid/moved 的旧位置；removed/moved 去重，nonrigid 不自动纳入。
2. 旧位置距最终 rescan GT 超过 0.05 m。
3. reference 阶段在有效深度中可见。
4. rescan 阶段至少两次观察到自由空间：3×3 深度邻域全部有效，且都在旧表面之后，裕量 `max(0.05 m, 0.01×观测深度)`；两次间隔≥5 帧，平移≥0.02 m 或旋转≥3°。

缺少可见/自由证据的已变化且最终未占据区域设为 neutral，不强行当 FN 或误删。无变化但未观察充分的区域不由这条规则一概豁免。评价阈值与检测器阈值分开管理，消融时评价域固定。

对每次真实擦除，用擦除前正权重 TSDF、擦除节点与六邻域恢复 incident-edge 零交叉；只计最后点云仍然缺失（距离>0.05 m）的表面。以 `sqrt(3)×0.025 m` 半径关联参考原子；所有动作取集合并集，重复删除不重复加分。

设 A 为进入 rescan 后实际擦除且最终缺失的非 neutral 参考原子，B 为此前的同类原子，E 为无法关联参考表面的最终缺失擦除原子（按网格去重）：

```text
TP = |A ∩ T|
FP = |(A ∪ B) \ T| + |(B ∩ T) \ A| + |E|
FN = |T| - TP
D  = TP / (TP + FP + FN)
P  = TP / (TP + FP)
R  = TP / (TP + FN)
```

提前擦除目标但 rescan 后没有有效动作的原子不获 TP，计入 FP，并仍是 FN；后来有效擦除可以建立该位置的最终正确结果。这个规则是协议选择，不是物理定律。分母为零时 D 约定为 0；P/R 分母为零时为 null。removed/moved 的分项诊断可能包含相同 FP，不能相加替代总体。

例如 T 有 100 个原子，正确清除 60，误清除 20，则 TP=60、FP=20、FN=40，D=0.5，P=0.75，R=0.6。只删得多不保证 D 高，重复删除也不加分。FP 是按这套几何证据判定的非目标动作，并不意味着每个 FP 都已经由人工证明是有害误删。

## 9. GAP、GAP25、GAP50

```text
GAP-v3 = AP   + (1 - AP)   × D
GAP50  = AP50 + (1 - AP50) × D
GAP25  = AP25 + (1 - AP25) × D
```

输入均为 0–1，小数计算后展示乘100。差值用百分点表示。当前没有独立 D25/D50：D 的原子审计与实例 IoU 阈值不同，因此两个 GAP 共用 D。

该复合分数等于 `1-(1-AP)(1-D)`，可解释为给剩余分数空间按 D 加分；不是推导得到的联合成功概率。D=0 时退化为 AP，D=1 时 GAP=1，即使 AP=0 也是如此。它能补充显式擦除能力，但可能掩盖分割失败，应与原始 AP、D、P/R 一起展示。

Baseline D=0 仅表示没有被该动作协议审计的显式擦除，不代表自然融合没有旧几何清理能力。若要比较最终残影，需另定义方法无关的结果评价，并控制初始重建覆盖。不能用 GAP 单项增益断言所有方面都改善。

## 10. 全量合并 PR 与场景均值

每种方法创建一个**完整、显式**目录清单，相对于 manifest 所在目录或使用绝对路径。例如在 WORK 下保存 `current-evaluations.json`：

```json
[
  "evaluation/current/val_095821f7_095821f9"
]
```

```bash
python tools/current_system/metrics.py pooled \
  --manifest "$WORK/current-evaluations.json" --out "$WORK/summary/current"
```

对 Baseline 用另一个清单和输出目录。程序检查重复场景、方法名、协议、评价器/核心/匹配 SHA；任何缺文件直接失败，不静默跳过。上例只含 1 对，不是全量报告；全量必须列出固定 110 对，并核对两方法的场景集合与逐场景 domain SHA 一致。

`pooled.json` 分列 `pooled_AP` 与 `scene_macro_mean`。前者调用原 AP 归约函数，直接合并所有匹配记录的 PR；后者先逐场景算 GAP 再平均，不能用 pooled AP 和 mean D 混算“总 GAP”。一般 `mean(AP+(1-AP)D)` 也不等于 `mean(AP)+(1-mean(AP))*mean(D)`，因为 AP 与 D 有相关性。P/R 等的有效场景数同时保存。

已完成历史全量结果可作为核对参考：110 对、47 组，pooled AP Baseline 6.577581% / Current 7.083715%；场景均值 AP 8.562004% / 9.345515%；Current mean D 9.125111%，mean GAP-v3 17.580494%。这些是既有完整运行结果，**不是本次发布重新跑出的 GPU 结果**，也不是新消融结果。

## 11. 验证与数据保留

```bash
python tools/current_system/test_metrics.py
python -m compileall -q tools/current_system
python tools/current_system/metrics.py --help
python tools/current_system/run_current.py --help
python tools/current_system/verify_release.py --out /absolute/path/core_check.json
```

8 项 CPU 测试覆盖无动作、正确/错误/提前/重复擦除、保留邻居零交叉、同帧不同候选归档搬迁、无目标与未定义分母、GAP 边界、IoU 严格阈值、pooled AP 不等于场景均值、混用协议/重复场景拒绝和固定清单数量。发布另核对已有 110 对的 GAP/D 公式与记录，结果见 [release_validation.json](release_validation.json)，但不以此冒充原始几何匹配的端到端复跑。本分支完整 CUDA 流水线尚需在配置好的实验环境做一次 smoke 验证。

保留每次运行的配置、源码 SHA、输入/masks 摘要、日志、最终点云/checkpoint、AP matches、GAP domain、擦除及 incident-edge 证据。准备/下载失败耗时与成功运行时间分列；不能为了表格好看删除 AP=0/D=0 的场景。使用不同目录保存旧类别筛选协议、新全实例协议和消融结果。具体实验矩阵与统计计划见 [消融plan.md](消融plan.md)。
