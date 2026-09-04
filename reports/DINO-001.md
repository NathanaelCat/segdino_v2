# DINO-001：SegDINO-v2 余弦退火学习率调度干预实验

> 状态：`completed`（已完成）
> 记录更新：2026-09-04
> 相关基线：`runs/osd_v2_current_frozen512_full50e_seed20260901` (Constant LR Baseline)

---

## 1. 实验元数据

| 字段 | 内容 |
|---|---|
| **实验编号** | `DINO-001` |
| **目标** | 探究在 SegDINO-v2 (DINOv3-S frozen) 上，学习率调度干预（Cosine 退火至 1e-6）对训练后半程（30~50 epoch）checkpoint 波动性的影响 |
| **研究假设** | 学习率步长干预能够显著改变后期的 checkpoint sensitivity；将恒定 1e-4 退火至 1e-6 可收窄相邻 epoch 间的性能方差 |
| **单一变量** | `scheduler` 由 `constant` 改为 `cosine`（3 epoch warmup 至 1e-4，随后余弦退火至 1e-6）；补充全 Epoch 平均 Loss 与标准 `train.log` |
| **对照基线** | `runs/osd_v2_current_frozen512_full50e_seed20260901` (Constant LR 1e-4, 50 epoch, Val Best 92.70%, Test@Val-best 90.96%) |
| **协议标准** | `OSD-EXP-v1.0/L4-global-512`；train/val/test=811/203/254；主指标 `mIoU3_report_only_global` |
| **配置文件** | `segdino_v2/configs/osd_v2_512_cosine.json` |
| **随机种子** | `20260901`（与基线一致，单 seed） |
| **训练长度** | 50 Epochs / 5100 Steps (Batch 8)；总耗时 1384.07 s (23.07 min) |
| **环境硬件** | Linux `vp-166`；NVIDIA GeForce RTX 4090 GPU0；Conda `segdino_osd` (PyTorch 2.2.0+cu118) |

---

## 2. 核心实验对比数据（Constant vs Cosine）

### 2.1 后半程波动性指标（Epochs 31~50）

| 统计指标 | Constant LR 基线 | Cosine 退火 (DINO-001) | 变化 |
|---|---:|---:|:---:|
| **后 20 轮标准差 (Std Dev)** | **1.5261 pp** | **0.5383 pp** | **方差降低 64.7%** |
| **后 20 轮平均相邻跳变** | **1.4494 pp** | **0.5503 pp** | **收窄 62.0%** |
| **后 20 轮最大相邻跳变** | **4.9116 pp** (E49→E50) | **1.8895 pp** | **跳变幅度收窄** |
| **后 20 轮最低波谷** | 87.3794% | 88.5686% | +1.19 pp |
| **Best 与 Latest 差值** | 0.00 pp (E50恰好最高) | 0.85 pp (E38最高 90.63% vs E50 89.78%) | 末期贴近 |

### 2.2 后期水平与分布数据

*注：序列内部存在高度自相关，均值微弱差异不作为模型优劣的充分统计证明，仅供水平参考。*

| 统计量 | Constant LR 基线 | Cosine 退火 (DINO-001) | 差异 |
|---|---:|---:|:---:|
| **后 20 轮平均 Val mIoU (31~50)** | 89.6004% | 89.6834% | +0.08 pp（基本相当） |
| **后 10 轮平均 Val mIoU (41~50)** | 89.5886% | 89.7095% | +0.12 pp（基本相当） |
| **后 10 轮标准差 (Last-10 Std)** | 1.6369 pp | **0.3702 pp** | 波动显著收紧 |
| **后 20 轮中位数 Val mIoU** | 89.4034% | 89.6952% | +0.29 pp |

### 2.3 关键评测点指标（L4-global-512 口径，254 张 Test）

| 评测项 | Constant LR 基线 | Cosine 退火 (DINO-001) | 行业 SOTA 参照 (SegFormer/UPerNet) |
|---|---:|---:|---|
| **Val mIoU3 (Best)** | **92.7026%** (Epoch 50) | **90.6276%** (Epoch 38) | - |
| • Oil IoU | 91.6741% | 88.9345% | - |
| • Water IoU | 89.1624% | 85.7616% | - |
| • Others IoU | 97.2712% | 97.1866% | - |
| **Val mIoU3 (Latest E50)** | 92.7026% | **89.7779%** | - |
| **Test mIoU3 (Test@Val-best)** | **90.9612%** | **89.9481%** | **超越 SegFormer-B5 (89.63%)、B0 (89.30%)** |
| • Oil IoU (Test) | 89.1774% | 87.6838% | UPerNet 为 87.75% |
| • Water IoU (Test) | 86.5287% | 85.0988% | UPerNet 为 86.97% |
| • Others IoU (Test) | 97.1774% | 97.0615% | UPerNet 为 96.65% |
| **Test mIoU3 (Test@Val-latest)** | 90.9612% | **90.0346%** | - |

---

## 3. 严格科学结论与边界

基于本次单 seed 对照实验，明确以下确凿结论与未解假设：

### 3.1 当前已确证的事实（Confirmed Evidence）
1. **调度干预对稳定性的影响极强（Strong Evidence）**：将学习率按余弦调度衰减至 1e-6，显著降低了后 20 轮的 checkpoint-to-checkpoint 方差（Std 从 1.53 pp 降至 0.54 pp，Last-10 Std 从 1.64 pp 降至 0.37 pp）。
2. **基线的高敏感性（High Checkpoint Sensitivity）**：Constant LR 在后期的波动极大，单次运行中第 49 轮（87.79%）到第 50 轮（92.70%）单轮跳变达 4.91 pp，说明 Constant 单次运行产生的 Best Checkpoint 具有极高的敏感性，必须进行多 seed 检验。
3. **单次峰值指标事实（Factual Metric Gap）**：在本次单 seed 运行中，Cosine 的 Val Best 比 Constant 低 2.07 pp，Test@Val-best 低约 1.01 pp。

### 3.2 待验证机制假设（Open Hypotheses for Further Testing）
1. **假设 1（剂量—响应关系）**：后期波动与学习率步长正相关，但退火下限设定过低（1e-6）可能限制了模型晚期探索更高精度的能力。
   - *检验手段*：进行 `min_lr` 梯度扫描（Cosine floor causal sweep: 1e-6, 1e-5, 2e-5, 5e-5, 1e-4），观察方差（Last-10 Std）与最优性能（Best Val, Test）的剂量响应曲线。
2. **假设 2（Constant 92.70% 的可复现性）**：Constant 基线在第 50 轮达到的 92.70% / Test 90.96%，到底是该策略下的可稳健复现水平，还是单 seed 偶发的高敏感性极值。
   - *检验手段*：对最有希望的调度配置与 Constant 基线运行 3 seeds 对比，统计真实的 Mean ± Std。

---

## 4. 下一步行动计划

不引入未经验证的复杂技巧（如直接上 EMA），严格按照因果干预原则推进：
1. **启动 Cosine Floor Causal Sweep**：在冻结骨干网络配置下，系统评测 `min_lr = [1e-6, 1e-5, 2e-5, 5e-5]`。每个配置严格记录 **Best Val、Last-10 Mean、Last-10 Std、Test@Val-best**。
2. **多 Seed 最终裁决**：选取 1~2 个表现最优的 scheduler 与 Constant 进行 3-seed 对齐评测，彻底回答稳定性与性能上限的权衡问题。
