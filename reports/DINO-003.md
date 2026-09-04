# DINO-003：SegDINO-v2 Cosine Floor 剂量—响应扫描实验（min_lr = 2e-5）

> 状态：`completed`（已完成）
> 记录更新：2026-09-04
> 相关基准：`DINO-001` (min_lr=1e-6) 与 Constant LR 基线 (1e-4)

---

## 1. 实验元数据

| 字段 | 内容 |
|---|---|
| **实验编号** | `DINO-003` |
| **目标** | 作为 Cosine Floor 剂量—响应实验的关键节点，验证将退火下界放宽至 `min_lr = 2e-5`（初始 LR 的 20%）对检查点敏感性（Last-10 Std）与测试集泛化性能（Test@Val-best）的影响 |
| **单一变量** | `scheduler.min_lr` 设为 `2e-5`（DINO-001 为 1e-6，基线为 Constant 1e-4）；其余所有模型架构、数据增强、损失、种子完全一致 |
| **对照基线** | Constant LR (1e-4) 与 DINO-001 (1e-6) |
| **协议标准** | `OSD-EXP-v1.0/L4-global-512`；train/val/test=811/203/254；主指标 `mIoU3_report_only_global` |
| **配置文件** | `segdino_v2/configs/osd_v2_512_cosine_floor2e5.json` |
| **随机种子** | `20260901` |
| **训练长度** | 50 Epochs / 5100 Steps；耗时 1391.28 s (23.19 min)；显存 ~15,081 MiB |
| **环境硬件** | Linux `vp-166`；NVIDIA GeForce RTX 4090 GPU0；Conda `segdino_osd` |

---

## 2. 剂量—响应三方对照表（Dose-Response Table）

按照本地端制定的严格评测标准，固定报告 **Best Val、Last-10 Mean、Last-10 Std、Test@Val-best**：

| 实验配置 | min_lr | Best Val | Last-10 Mean | Last-10 Std (波动性) | Test@Val-best (254张Test) |
|---|---|---:|---:|---:|---:|
| **Constant 基线** | 1e-4 (无衰减) | **92.7026%** (E50) | 89.5886% | **1.6369 pp** (高波动) | **90.9612%** |
| **DINO-003 (本实验)** | **2e-5** (适度衰减) | **91.2620%** (E50) | **89.7778%** | **0.8787 pp** (中低波动) | **90.3798%** |
| **DINO-001** | 1e-6 (深度衰减) | 90.6276% (E38) | 89.7095% | **0.3702 pp** (超低波动) | 89.9481% |

### 关键评测细分（DINO-003 Test@Val-best）
- **Test mIoU3**: **90.3798%**
  - • Oil IoU: **88.2155%**
  - • Water IoU: **85.8625%**
  - • Others IoU: **97.0613%**

---

## 3. 严格科学观察

1. **单调趋势证据（Monotonic Trend Supporting the Hypothesis）**：
   - 在当前单 seed 运行下，随 `min_lr` 从 1e-4 $\to$ 2e-5 $\to$ 1e-6，后 10 轮标准差呈现清晰的单调递减趋势（`1.64 pp` $\to$ `0.88 pp` $\to$ `0.37 pp`）；
   - 为“学习率调度显著影响训练后半程检查点敏感性（Checkpoint Sensitivity）”的假设提供了强有力的初步实验支持。
2. **后期性能期望的一致性（Invariance of Mean Performance）**：
   - 三个配置的 Last-10 均值分别为 89.59%、89.78%、89.71%，差异均在 0.2 pp 以内，表明调度策略主要影响检查点围绕均值的方差分布，而非改变整体水平。

---

## 4. 实验规范声明（Test Set Usage Discipline）

- **严禁利用测试集调参**：测试集（254 张 Test）严格作为各配置决策完成后的独立一次性报告，绝不用于反向指导学习率或调度器下限的选择；
- 后续补齐 `min_lr = 1e-5` 和 `5e-5` 扫描时，模型的筛选与收敛评判将**严格且仅基于 Train + Val 指标**（Best Val、Last-10 Mean、Last-10 Std），锁定最优策略后再进行最终的测试集报告与 3-seed 统计检验。
