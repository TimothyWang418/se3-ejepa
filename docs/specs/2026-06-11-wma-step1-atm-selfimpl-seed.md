# wma-step1 seed：ATM 自实现 + LeWM/PushT 对拍（预注册）

- **日期**：2026-06-11　**所属**：P4-D / WMA（The World Model Audit），启动第一刀
- **上游**：`papers/proposals/p4d-world-model-audit-launch.md`；ATM = arXiv:2606.09028（官方代码截至 2026-06-11 未释出，README 占位仓复查确认）
- **目标**：把 ATM 交叉矩阵做成本 repo 第 II 件审计仪器（可复用模块），先在**全合成系统上认证仪器本身**，再对 LeWM/PushT 官方 checkpoint 出第一列报告卡，与 ATM 论文 Table 6 LeWM/PushT 行对拍。
- **scope guard**：本步不做 $D_{i,j}(g)$ 等变升级（wma-step3）、不做筛选分 $S_{\mathrm{ATM}}$ 的 $\lambda$ 拟合（避免"免仿真"的鸡生蛋——只报组件量）、不扫多模型（wma-step2）。

## 协议（冻结）

1. **特征**（ATM 公式 4）：$\xi_t^T = [z_t,\, z_{t+1},\, z_{t+1}-z_t]$，$\xi_t^P = [z_t,\, \hat z_{t+1},\, \hat z_{t+1}-z_t]$；token/patch 级 latent 先 mean-pool。
2. **探针**（公式 5）：$h(\xi) = W_2\,\mathrm{GELU}(\mathrm{LN}(W_1\xi + b_1)) + b_2$。论文未给隐宽/训练超参 → **容量 mini-scan：隐宽 $\{64, 256, 1024\}$**（主报告 256，三点敏感性随结果披露——ATM 局限 1 的协议级缓解）。Adam lr $10^{-3}$、batch 256、≤2000 步、10% val 早停（patience 5×50 步）。fresh probe per (model, domain, width, seed)；probe seed ∈ {0,1,2}。
3. **读出**：$D_{i,j} = \mathbb{E}\|h_i(\xi^j) - a_t\|_2^2$（公式 6）→ $G_{T\to P}$、$G_{P\to T}$、$I_{\mathrm{diag}}$、$L_{\mathrm{ATM\text{-}sym}}$（公式 8–9，$\epsilon = 10^{-8}$）。**新增（本工具的协议改进）**：归一化 $\tilde D_{i,j} = D_{i,j} / \mathrm{Var}(a)$（$= 1 - R^2$ 形式，尺度自由）——跨模型报告卡只用 $\tilde D$；这是对 ATM Table 4 跨族系数迁移失败（68.54%）的协议级回应。
4. **数据**：PushT 转移 $(o_t, a_t, o_{t+1})$。主路径 = vendored stable-worldmodel PushT env + **seeded 行为策略**（uniform-random 动作或 WeakPolicy，先跑通者为准，披露）、$N = 5000$ 转移、train/val/test = 70/10/20（按 episode 切分防泄漏）；若 swm 官方离线集可直接加载则优先官方集（披露用了哪个）。env seed=0 固定。
5. **模型**：`models/lewm/pusht-weights.pt` + config，**复用 step91 的 strict-load 路径**（`experiments/step91_lewm_audit.py::load_lewm` 同源逻辑），encoder/predictor 全程冻结。

## 预注册 gates

| Gate | 设置 | 判据 |
|------|------|------|
| **G1a 完美预测器**（合成：$z' = Az + Ba + \sigma\eta$，$F$ = 真映射）| 仪器在理想域内 | $\lvert G_{T\to P}\rvert < 0.05$ 且 $L_{\mathrm{sym}} < 0.2$ 且 $\tilde D_{T,T} < 0.1$ |
| **G1b 破坏预测器**（$F$ = 与 $(z,a)$ 无关的冻结噪声）| 坏预测器必须被抓 | $\tilde D_{P,P} \ge 0.7$ 且 $D_{T,P}/D_{T,T} \ge 5$ |
| **G1c 捷径注入**（$F$ 把 $a$ 原文写进 latent 末坐标；$T$ 域无此通道）| 复现 AITS-P 病理签名 | $L_{\mathrm{sym}} \ge 10\times$ G1a 值，且形态 = $D_{P,P}$ 塌、$D_{P,T}$ 高（域专属动作码） |
| **G2 LeWM/PushT 健康域对拍**（pattern 门，**非数值复现门**）| ATM Table 6 LeWM/PushT 行：$D_{T,T} = 0.183$、$\lvert G\rvert = 0.037$、$L_{\mathrm{sym}} = 0.226$（raw）；病理行 $L_{\mathrm{sym}} \ge 3$ | 我方数字落**健康域**：$\lvert G_{T\to P}\rvert \le 0.5$ 且 $L_{\mathrm{sym}} \le 2.0$（阈值取健康/病理两簇之间的空隙）。$D_{T,T}$ 量级偏离系数**如实披露不设硬门**（数据分布 / probe 超参 / 动作归一化三处自由度论文未给）|
| **G3 infra parity** | 与 step91 共底座 | latent 维度与 encoder 行为和 step91 记录一致；predictor 一步误差量级与 step91 相容 |

**fallback（预注册）**：G2 若 FAIL → 先查动作归一化与数据分布两个已知自由度；复查后仍 FAIL 则如实记录，并把"ATM 数字对实现细节的敏感性"本身升格为发现（工具文合法内容：复现性审计）。

## 产出

- `src/audit/atm.py`（可复用仪器模块）+ `experiments/wma_step1_atm_probe.py`（实验脚本）
- `tests/test_wma_step1.py`（G1a–c 全合成门做成单测；探针/矩阵/读出的数学正确性测试）
- `papers/figures/wma_step1_atm_lewm.json` + `papers/wma_record.md` 记账

## 红线

探针容量披露随结果走；不碰 `data/p4_step1/`（paper3 活跃区）；不改 paper2 文件。
