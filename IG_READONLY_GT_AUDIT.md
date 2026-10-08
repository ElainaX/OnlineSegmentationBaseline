# IG 只读 would_remove 与 3RScan GT 对照（2026-10-08）

20条确认候选中，14条有较强GT变化支持，3条混入其他实例，3条没有官方变化标注支持。这里判断的是候选与“发生变化的物体”的关联，不等于证明整个实例应该删除。没有执行实际删除，没有重新计算AP，也没有修改IG/IT的算法配置。

## 如何判断

将确认时保存的冻结参考点、累计矛盾点分别关联到最近的reference GT顶点，主容差5厘米。最近邻在全部GT顶点中查询，而不是只在变化物体中查询。比例的分母包含该区域所有点，包括未匹配及无实例标签的点。GT和预测使用已经对齐的reference世界坐标，不再次施加变换。

“变化支持较强”表示整个冻结候选和累计矛盾点各自至少80%对应变化标注实例；“混合”表示累计矛盾点至少50%对应变化实例，但未达到上述条件。这是诊断分组，不是官方precision、AP或经过调参验证的删除阈值。刚体移动、移除、非刚体变化分别保留；非刚体变化不能解释成整个物体消失。

四个pair的prepared session_boundary与原始3RScan.json中的rigid、removed、nonrigid逐项完全一致。对象名称来自对应原始semseg.v2.json。all-things标签中的0属于未提供实例标签的表面（包括结构类），不能当作确定静态的反例。

## 全部候选

C编号是各pair的candidate_id；GT编号是原始objectId。百分比对应任意变化标注实例的总和，而非仅表中占比最大的GT对象。确认帧是实验输出的frame ID。

| Pair | 候选 | 确认帧 | 矛盾点主要GT对象 | 该对象变化标注 | 整候选变化占比5cm | 矛盾点变化占比5cm | 矛盾点变化占比10cm | 判断 |
|---|---|---:|---|---|---:|---:|---:|---|
| 01 | C2 | 288 | GT3 table | 移除 | 84.0% | 95.3% | 99.6% | 变化支持较强 |
| 01 | C3 | 374 | GT24 pillow | 非刚体变化、刚体移动 | 95.3% | 99.5% | 100.0% | 变化支持较强 |
| 02 | C4 | 1085 | GT7 bedside table | 无变化标注 | 0.0% | 0.0% | 0.0% | 无变化标注支持 |
| 02 | C6 | 1155 | GT15 blanket | 非刚体变化 | 70.1% | 78.1% | 78.1% | 混合，不能整候选删除 |
| 02 | C7 | 1535 | GT4 clothes | 非刚体变化 | 97.2% | 98.4% | 98.9% | 变化支持较强 |
| 02 | C9 | 1445 | GT38 curtain | 非刚体变化 | 88.1% | 82.0% | 94.9% | 变化支持较强 |
| 02 | C12 | 1325 | GT23 bag | 无变化标注 | 0.0% | 0.0% | 0.0% | 无变化标注支持 |
| 02 | C15 | 1450 | GT38 curtain | 非刚体变化 | 97.5% | 99.3% | 99.3% | 变化支持较强 |
| 02 | C16 | 1480 | GT38 curtain | 非刚体变化 | 93.8% | 91.4% | 97.7% | 变化支持较强 |
| 03 | C1 | 570 | GT28 chair | 无变化标注 | 0.0% | 0.0% | 0.0% | 无变化标注支持 |
| 03 | C5 | 400 | GT35 box | 移除 | 90.9% | 89.0% | 95.1% | 变化支持较强 |
| 03 | C7 | 488 | GT11 chair | 刚体移动 | 98.3% | 97.6% | 100.0% | 变化支持较强 |
| 03 | C9 | 570 | GT9 chair | 刚体移动 | 47.8% | 61.8% | 63.4% | 混合，不能整候选删除 |
| 03 | C10 | 492 | GT10 chair | 刚体移动 | 71.3% | 76.2% | 77.3% | 混合，不能整候选删除 |
| 03 | C12 | 548 | GT22 item | 移除 | 95.3% | 96.4% | 98.2% | 变化支持较强 |
| 04 | C4 | 456 | GT4 chair | 刚体移动 | 99.0% | 98.6% | 100.0% | 变化支持较强 |
| 04 | C6 | 568 | GT40 chair | 刚体移动 | 98.9% | 99.2% | 100.0% | 变化支持较强 |
| 04 | C7 | 456 | GT29 chair | 移除 | 95.7% | 95.5% | 100.0% | 变化支持较强 |
| 04 | C9 | 540 | GT5 chair | 刚体移动 | 95.9% | 96.7% | 99.2% | 变化支持较强 |
| 04 | C12 | 532 | GT17 chair | 刚体移动 | 97.9% | 98.6% | 100.0% | 变化支持较强 |

## 需要限制删除范围的3条

- pair02 C6：78.1%矛盾点对应GT15毯子（非刚体变化），19.1%对应未标注变化的GT10床。整个候选的变化占比仅70.1%。
- pair03 C9：61.8%矛盾点对应GT9移动椅子，29.4%对应未标注变化的GT8桌子。整个候选的变化占比仅47.8%。
- pair03 C10：74.5%对应GT10移动椅子，1.7%对应GT11移动椅子，19.9%对应GT8桌子；变化合计76.2%，整个候选为71.3%。

这些候选确实命中了变化物体，但候选内部已混入其他实例。支持“局部矛盾合理”，不支持“整条track安全删除”。

## 没有官方变化标注支持的3条

- pair02 C4：89.9%矛盾点对应GT7床头柜，变化占比0。reference/rescan同ID GT双向最近邻距离中位数为4.7/4.3厘米，大部分几何仍然重叠（双向10厘米内比例85.4%/84.9%）；矛盾点集中在不重合边侧，应列为疑似误报，但尚未确定是边界、覆盖、配准还是深度原因。
- pair02 C12：98.9%矛盾点对应GT23包，变化占比0。同ID GT双向最近邻距离中位数为8.1/6.1厘米，双向10厘米内比例66.7%/84.6%。有表面偏差，但没有GT支持将其视为应删除物体。
- pair03 C1：99.6%矛盾点对应GT28椅子，变化占比0。不过同ID GT在两次扫描中的几何明显不同：双向最近邻距离中位数22.0/43.4厘米，双向10厘米内比例仅7.8%/5.9%；矛盾点到rescan同ID表面的距离中位数32.6厘米。这是变化列表与同ID几何之间的不一致，不能直接判为确定误报，也不能确定是漏标；还需区分漏标、实例对应及局部重建/对齐问题。质心差55.4厘米仅用于描述两点集，不能当作物体真实位移量。

## 不应夸大的结论

1. pair02 C9/C15/C16都主要命中GT38窗帘，是同一物体的三个候选。17条变化相关候选（14强+3混合）对应15个主要变化对象，而不是17个独立物体。
2. pair01 C3主要对应GT24枕头，该ID同时有刚体与非刚体标注。累计矛盾点中98.2%属于GT24，另1.4%属于GT35毯子，因此表中99.5%不是单一枕头的纯度。
3. pair04 C9累计矛盾点94.2%属于GT5椅子、2.5%属于GT4椅子，表中96.7%也是变化支持总和。
4. pair04 C6虽对应移动椅子，但16.2%矛盾点距rescan GT仍在5厘米内。发生移动不意味着每一个旧表面点都应删除。一般而言，rescan GT缺少邻近表面也不能独立证明观测到了自由空间。
5. 本轮没有统计未触发would_remove的GT变化对象，因此不能报告召回率；不能把上述分组直接称为最终删除precision。

## 匹配容差敏感性

| 最近邻容差 | 强支持 | 混合 | 未达到50%变化支持 |
|---|---:|---:|---:|
| 2.5 cm | 0 | 10 | 10 |
| 5 cm | 14 | 3 | 3 |
| 10 cm | 14 | 3 | 3 |

5和10厘米下候选分组逐条一致；2.5厘米下大量点未匹配，不能将其自动解释为物体无变化。这说明绝对点比例受重建表面偏差、采样和容差影响。5厘米是本次离线对照口径，不是已验证的通用最优阈值。

## 几何与原始检测图

以下几何图红色为reference GT，蓝色为rescan同ID GT，黑色为累计矛盾点；三个面板为XY/XZ/YZ投影，单位米。

### pair02_C4

![pair02_C4 GT几何对照](<E:/Code/github/ig_readonly_20261008/gt_audit/pair02_C4_geometry.png>)

### pair02_C6

![pair02_C6 GT几何对照](<E:/Code/github/ig_readonly_20261008/gt_audit/pair02_C6_geometry.png>)

### pair02_C12

![pair02_C12 GT几何对照](<E:/Code/github/ig_readonly_20261008/gt_audit/pair02_C12_geometry.png>)

### pair03_C1

![pair03_C1 GT几何对照](<E:/Code/github/ig_readonly_20261008/gt_audit/pair03_C1_geometry.png>)

### pair03_C9

![pair03_C9 GT几何对照](<E:/Code/github/ig_readonly_20261008/gt_audit/pair03_C9_geometry.png>)

### pair03_C10

![pair03_C10 GT几何对照](<E:/Code/github/ig_readonly_20261008/gt_audit/pair03_C10_geometry.png>)

### 确认时六面板检测图

- [pair_01_0cac767c_19f1a892 C2 frame 288](<E:/Code/github/ig_readonly_20261008/results/pair_01_0cac767c_19f1a892/remove_detection/frames/000288.png>)
- [pair_01_0cac767c_19f1a892 C3 frame 374](<E:/Code/github/ig_readonly_20261008/results/pair_01_0cac767c_19f1a892/remove_detection/frames/000374.png>)
- [pair_02_751a55a1_751a557f C4 frame 1085](<E:/Code/github/ig_readonly_20261008/results/pair_02_751a55a1_751a557f/remove_detection/frames/001085.png>)
- [pair_02_751a55a1_751a557f C6 frame 1155](<E:/Code/github/ig_readonly_20261008/results/pair_02_751a55a1_751a557f/remove_detection/frames/001155.png>)
- [pair_02_751a55a1_751a557f C7 frame 1535](<E:/Code/github/ig_readonly_20261008/results/pair_02_751a55a1_751a557f/remove_detection/frames/001535.png>)
- [pair_02_751a55a1_751a557f C9 frame 1445](<E:/Code/github/ig_readonly_20261008/results/pair_02_751a55a1_751a557f/remove_detection/frames/001445.png>)
- [pair_02_751a55a1_751a557f C12 frame 1325](<E:/Code/github/ig_readonly_20261008/results/pair_02_751a55a1_751a557f/remove_detection/frames/001325.png>)
- [pair_02_751a55a1_751a557f C15 frame 1450](<E:/Code/github/ig_readonly_20261008/results/pair_02_751a55a1_751a557f/remove_detection/frames/001450.png>)
- [pair_02_751a55a1_751a557f C16 frame 1480](<E:/Code/github/ig_readonly_20261008/results/pair_02_751a55a1_751a557f/remove_detection/frames/001480.png>)
- [pair_03_f62fd5f8_20c9939d C1 frame 570](<E:/Code/github/ig_readonly_20261008/results/pair_03_f62fd5f8_20c9939d/remove_detection/frames/000570.png>)
- [pair_03_f62fd5f8_20c9939d C5 frame 400](<E:/Code/github/ig_readonly_20261008/results/pair_03_f62fd5f8_20c9939d/remove_detection/frames/000400.png>)
- [pair_03_f62fd5f8_20c9939d C7 frame 488](<E:/Code/github/ig_readonly_20261008/results/pair_03_f62fd5f8_20c9939d/remove_detection/frames/000488.png>)
- [pair_03_f62fd5f8_20c9939d C9 frame 570](<E:/Code/github/ig_readonly_20261008/results/pair_03_f62fd5f8_20c9939d/remove_detection/frames/000570.png>)
- [pair_03_f62fd5f8_20c9939d C10 frame 492](<E:/Code/github/ig_readonly_20261008/results/pair_03_f62fd5f8_20c9939d/remove_detection/frames/000492.png>)
- [pair_03_f62fd5f8_20c9939d C12 frame 548](<E:/Code/github/ig_readonly_20261008/results/pair_03_f62fd5f8_20c9939d/remove_detection/frames/000548.png>)
- [pair_04_752cc595_55551083 C4 frame 456](<E:/Code/github/ig_readonly_20261008/results/pair_04_752cc595_55551083/remove_detection/frames/000456.png>)
- [pair_04_752cc595_55551083 C6 frame 568](<E:/Code/github/ig_readonly_20261008/results/pair_04_752cc595_55551083/remove_detection/frames/000568.png>)
- [pair_04_752cc595_55551083 C7 frame 456](<E:/Code/github/ig_readonly_20261008/results/pair_04_752cc595_55551083/remove_detection/frames/000456.png>)
- [pair_04_752cc595_55551083 C9 frame 540](<E:/Code/github/ig_readonly_20261008/results/pair_04_752cc595_55551083/remove_detection/frames/000540.png>)
- [pair_04_752cc595_55551083 C12 frame 532](<E:/Code/github/ig_readonly_20261008/results/pair_04_752cc595_55551083/remove_detection/frames/000532.png>)

## 可复核文件

- [完整逐候选JSON](<E:/Code/github/ig_readonly_20261008/gt_audit/candidate_gt_audit.json>)
- [离线审计脚本](<E:/Code/github/ig_readonly_20261008/port/audit_readonly_against_gt.py>)
- [本轮原始运行记录](<E:/Code/github/ig_readonly_20261008/IG_READONLY_RESULTS.md>)
- 远端审计结果：`/root/autodl-tmp/output/ig_readonly_20261008/gt_audit/`。pair04采用diagnostic_analysis.json指定的成功重试输出，未使用未完成的首次运行。
- 原始变化标注SHA256：`674a00f50f76b198b9de44efd86c390fea3da37ba8f12cf8ccd00045e265fa64`。
