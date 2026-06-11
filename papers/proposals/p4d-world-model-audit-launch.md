# P4-D / WMA：The World Model Audit——启动书（proposal v0.1）

- **日期**：2026-06-11
- **来源**：vault 启动包 `raw/papers/研究线-paper4-候选地图与启动包.md`（候选 P4-D）
- **状态**：novelty 扫描完成（SAFE），wma-step1 seed 已注册
- **定位**：工具文 / 副业（paper3 等训练窗口期后台推进，3-6 周占旗）
- **venue**：NeurIPS/ICLR workshop 或 Datasets & Benchmarks track + arXiv + 公开报告卡页

---

## 0. 命名决定（防撞车）

paper3 的实验步骤前缀是 `P4-step*`（P4 = 研究线 Phase 4，**不是** paper4）。为零歧义，本项目在 repo 内一律用 **`wma`**（World Model Audit）前缀：`experiments/wma_step*_*.py`、`tests/test_wma_step*.py`、`papers/wma_record.md`、`papers/figures/wma_step*_*.json`。vault 侧沿用 "P4-D" 称呼，二者同指。

## 1. 使命（一句话）

给预训练世界模型 checkpoint 做**零训练部署体检**：四件正交仪器一条命令，输出报告卡——**能信几步**（谱/horizon 证书）× **动作语义在不在**（ATM 交叉矩阵）× **逐维结构懂不懂**（PoR 平移/旋转/夹爪拆解）× **等变性有多少**（commutant / 群作用一致性）。

## 2. 为什么现在（科学理由，不只是工程打包）

1. **大厂排队放 checkpoint**（hook 5）：TD-MPC2 / LeWM / V-JEPA 2-AC 已审过；DreamZero-DROID 14B（GEAR）2026 上 HF；每个新 checkpoint = 零训练免费实验。
2. **step98/99 头条已证明单仪器会过度承诺**：纯谱审计给 V-JEPA 2-AC 发 $T_1 \approx 9$，被 20 条真机 DROID episodes 推翻（实测斜率 $0.033 \ll \lambda_1 = 0.178$，从未进入线性化邻域）。**交叉验证的审计才是可部署对象**——这正是把多件仪器拼成一个面板的定理级动机。
3. **下游用户零保护**："拿预训练 WM 部署"的人目前没有任何免仿真避坑工具；体检发现自带新闻价值（V-JEPA 2-AC 过度承诺实锤已是案例）。
4. 占领 "world model audit" 术语，给 paper2/3 滚引用。

## 3. Novelty 扫描（2026-06-11，verdict：SAFE）

| 邻居 | 测什么 | 差异化 |
|------|--------|--------|
| WorldOlympiad（arXiv:2606.11129，Alibaba DAMO，2026-06）| 生成长视频质量三项全能（物理/几何/交互保真）| 评**生成像素**；我们审计 **latent 机器**的部署可信度 |
| WorldModelBench / 4DWorldBench | video-gen-as-WM 质量评测 | 同上 |
| WorldArena 2.0（arXiv:2605.17912）| WM **作为环境**：perceptual quality / interactive utility / 跨平台 | 性能基准；我们是内部结构审计（谱/探针/等变），零训练 post-hoc |
| ATM（arXiv:2606.09028，单作者）| 仪器 II 的来源 | 代码**未释出**（GitHub README 占位，2026-06-11 复查）；我们：自实现首发 + 容量 mini-scan + 尺度归一化 + 并入面板 |
| PoR（arXiv:2606.07687，SNU）| 仪器 IV 的来源（encoder 侧逐维探针）| 本文自己不放代码；协议已在笔记复原 |
| Olaf-World（arXiv:2602.10104）| video latent action 线性探针、跨域协议 | 近邻仪器（latent action 侧），引用对照 |
| Geng（arXiv:2512.08991）| conformal WM 验证 | 统计校准 vs 解析证书 + 探针面板 |
| SCSA（arXiv:2605.22164）| 选择器审计程序 | "audit" 一词用于协议步骤，非工具/术语占用 |
| Guaranteed Safe AI（arXiv:2405.06624）| 呼吁 WM 应 auditable/monitorable at run time | **motivation 引用**：我们交付的正是这个呼吁的工具 |

**结论**：「world model audit」术语 + "部署可信度审计套件"定位无人占用。探针型诊断在升温（ATM / Olaf-World / PoR 三个月内三篇）——**窗口存在但在收窄**，3-6 周节奏合理。

## 4. 仪器面板（I–V）

| # | 仪器 | 测什么 | 现状 | 落地步骤 |
|---|------|--------|------|---------|
| I | 谱/horizon（`scripts/wm_audit.py` + `audit_map()`）| $\lambda_1$、$T_1(\epsilon)$、EXPANSIVE/ABSTAIN verdict | ✅ 生产级（paper2，3 家族 loader）| 已有 |
| II | ATM 交叉矩阵 | $D_{T,T}$、$\lvert G_{T\to P}\rvert$、$L_{\mathrm{sym}}$（动作语义跨域一致性）| ❌ 需自实现（官方代码未释出）| **wma-step1** |
| III | 等变精确度 | commutant 距离（step26 闭式探针）+ 轻量 $D_{i,j}(g)$（输入域群作用一致性——ATM 的等变升级，仪器首发、理论留给后续）| step26 有原型 | wma-step3 |
| IV | PoR 逐维探针 | 平移/旋转/夹爪逐维 $R^2$ | ❌ 协议在笔记里全须全尾 | wma-step4（7-DoF 数据上才有意义：V-JEPA 2-AC × DROID）|
| V（候选）| gap-mode（`src/audit/gap_mode.py`）| 可预测性差距 | ✅ paper3 仪器（3/3 认证门）| 是否入面板待 paper3 里程碑后定 |

## 5. 体检对象清单（第一刀产出之一）

| Tier | 对象 | 接口可行性 | 已有数据 |
|------|------|-----------|---------|
| **0（已有谱审计列）** | TD-MPC2 官方 15-cell（5 任务 × 3 seeds）+ MT 规模梯 1M–317M | ✅ loader 现成 | step89/92：校准 0.83–1.02；"scale does not buy a calibrated horizon" |
| 0 | LeWM PushT（quentinll/lewm-\*，MIT）| ✅ loader 现成（step91）| $\lambda_1 \approx 0$ ABSTAIN；step97 部署 cell |
| 0 | V-JEPA 2-AC ViT-g 1B（Meta 官方）| ✅ loader 现成（step98，SDPA workaround）| EXPANSIVE $T_1 \approx 9$ 过度承诺实锤 + DROID 真机监控（step99）|
| **1（GO，loader 路径清晰）** | DINO-WM（公开 ckpt）| 侦察：依官方 repo 加载 | ATM Table 4 有跨族参考数字可对拍 |
| 1 | LeWM 其它任务 ckpt | ✅ 同 step91 路径 | — |
| **2（侦察后定）** | DreamZero-DROID 14B（GEAR，HF，2026 新）| ⚠️ 视频扩散接口——仪器 I 需要 $C^1$ 确定性 latent 映射，可能只有 II/IV 适用 | 无 |
| 2 | DreamerV3 系 | ⚠️ RSSM 随机路径 + 官方 ckpt 稀缺 | 无 |
| 2 | Cosmos 系 | ⚠️ 扩散，仪器 I 不适配 | 无 |
| **X（排除）** | Genie 3 | ❌ 闭源权重 | — |

**报告卡矩阵目标**：≥6 模型 × ≥3 仪器；每格三态：数值 / **N/A（接口不适配，披露原因）** / 待跑。注意：N/A 本身是发现——"哪类架构根本无法被审计"是论文的一节（扩散视频 WM 连'能信几步'这个问题都无法被提出）。

## 6. 里程碑

- **wma-step1**（本周）：ATM 自实现（`src/audit/atm.py`）+ 合成认证门 G1a–c + LeWM/PushT 对拍 G2/G3 → 首列报告卡
- **wma-step2**：ATM 列扫 Tier 0（TD-MPC2 × 任务格、V-JEPA 2-AC × DROID）→ 首张双仪器报告卡（谱 × ATM 正交性的第一张实数据图）
- **wma-step3**：轻量 $D_{i,j}(g)$ 仪器首发 + commutant 列
- **wma-step4**：PoR 逐维列（V-JEPA 2-AC / DROID 7-DoF）
- **wma-step5**：统一 CLI（`wm_audit.py` 扩 `--mode panel`）+ 报告卡 leaderboard 页 + arXiv 工具文

## 7. 写作红线（启动包决策规则 3 继承）

- **所有探针类主张附容量披露**（Linear Probing 容量混淆）——wma-step1 起协议内置 3 宽度 mini-scan，不是事后补丁；
- 过框架陷阱 strong-reading 自查；
- 若沾 FEP 过 FEP 写作检查表（P4-D 预计不沾）；
- 对拍/复现类表述诚实：ATM 论文未给 probe 超参与数据分布——对拍门只设 **regime/pattern 级**，不设数值复现门。

## 8. 边界纪律

- 不动 paper2 文件（研究线会话领地）；不动 `data/p4_step1/`（paper3 活跃区）；
- 新文件全部 `wma` 前缀；预注册 gate 文化延续（spec 先行，诚实判定，负样本如实记录）。
