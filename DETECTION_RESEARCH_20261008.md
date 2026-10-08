# 变化检测论文与本轮实验对应关系（2026-10-08）

检索重点是 CVPR 2026、ICCV 2025 的主会论文。以下阅读了原文方法部分，不把论文摘要中的性能直接当成我们系统的性能。

| 工作 | 值得借鉴的机制 | 接入当前项目的边界 |
|---|---|---|
| [Changes in Real Time: Online Scene Change Detection with Multi-View Fusion，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Galappaththige_Changes_in_Real_Time_Online_Scene_Change_Detection_with_Multi-View_CVPR_2026_paper.html) | 把多视角变化线索保存在三维表示中，持续更新，而不是每帧独立二值决策。 | 依赖预建 3DGS，并在测试时优化变化参数。“不需要变化标签”不等于完全不做优化。本轮只借鉴空间记忆，没有复现其损失或 SAM2 特征。 |
| [Gaussian Mapping for Evolving Scenes（GaME），CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Yugay_Gaussian_Mapping_for_Evolving_Scenes_CVPR_2026_paper.html) | 用深度穿透和颜色差异寻找过期表面，再借助历史视角的分割掩码扩展更新范围。 | 我们测试深度与颜色的共同证据；当前实例可能污染，暂不能照搬“局部变化扩展为整个物体删除”。 |
| [CL-Splats，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Ackermann_CL-Splats_Continual_Learning_of_Gaussian_Splatting_with_Local_Optimization_ICCV_2025_paper.html) | 比较真实图像与旧场景渲染的 DINOv2 特征，将二维变化通过多视角投票落到三维区域。 | 它采用批次更新。本轮借鉴三维位置投票，没有安装或宣称复现 DINOv2/3DGS 流程。 |

原文方法：[在线多视角检测](https://arxiv.org/html/2511.12370v2)、[GaME](https://arxiv.org/html/2506.06909v2)、[CL-Splats](https://arxiv.org/html/2506.21117v1)。

本轮工程推导有三条，属于我们针对当前代码的判断，不是上述论文已经证明了本项目有效：

1. **证据必须存到具体表面上。** 看见椅背仍在，不应清掉椅腿区域以前的记录；没看见、被遮挡也不能当成“仍在”或“消失”。
2. **转头不等于从另一侧看。** 相机中心不变时，到固定三维点的射线不变；因此另测了按观察射线夹角选择独立证据。
3. **像素错位不等于空间消失。** 当前深度在旧点投影处较远，但附近仍有实测三维表面时，应该保守看待消失判断；另测两个体素半径的表面支持核验。

论文启发只提供可检验的设计方向。所有实际效果以 RESULTS.md 和保存的实验 JSON 为准；负面结果同样保留。
