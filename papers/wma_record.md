# WMA 台账（P4-D：The World Model Audit）

> **命名**：vault 启动包里的候选 "P4-D" = repo 里的 **WMA**（World Model Audit）。前缀 `wma` 防撞 paper3 的 `P4-step*`（Phase 4 ≠ paper4）。
> **文化**：继承 paper2/3——spec 先行、gate 预注册、诚实判定、负样本如实记录。
> **启动包**：vault `raw/papers/研究线-paper4-候选地图与启动包.md`；启动书：`papers/proposals/p4d-world-model-audit-launch.md`。

---

## [2026-06-11] wma-step0 | 启动：novelty 扫描 + 启动书 + step1 seed 注册

- **Novelty verdict：SAFE**——「world model audit」术语 + "部署可信度审计套件"定位无人占用。最近邻全部差异化清晰：WorldOlympiad（生成视频质量）、WorldArena 2.0（WM-as-environment 性能）、SCSA（"audit"用于协议步骤）；Guaranteed Safe AI（2405.06624）是 motivation 引用（呼吁 WM 可审计——我们交付工具）。细节表见启动书 §3。
- **ATM 官方代码复查：未释出**（GitHub README 占位仓，1 commit）→ wma-step1 自实现 = 首个公开可运行实现。
- **hook 5 命中**：DreamZero-DROID 14B（GEAR）已上 HF → 入体检对象清单 Tier 2（视频扩散接口，仪器 I 可能不适配——N/A 本身是发现）。
- **体检对象清单 v1**：Tier 0 = TD-MPC2 15-cell + MT 梯 / LeWM / V-JEPA 2-AC（谱列已有，补其余仪器列）；Tier 1 = DINO-WM（ATM Table 4 跨族数字可对拍）；Tier 2 = DreamZero / DreamerV3 / Cosmos（侦察后定）；排除 Genie 3（闭权重）。目标 ≥6 模型 × ≥3 仪器。
- **注册**：`docs/specs/2026-06-11-wma-step1-atm-selfimpl-seed.md`（G1a–c 合成认证 / G2 LeWM 对拍 pattern 门 / G3 infra parity；fallback 预注册）。
- **下一刀**：wma-step1 build——`src/audit/atm.py` + 合成门单测 + LeWM/PushT 跑数。

## [2026-06-11] wma-step1 | ATM 仪器自实现 + LeWM/PushT 首列：G1 3/3、G3 PASS；G2 FAIL → 烙印机制确认

- **仪器**：`src/audit/atm.py`——公式 4–10 自实现 + 两项注册协议改进（容量 mini-scan $\{64,256,1024\}$、方差归一化 $\tilde D = D/\mathbb{E}\|a-\bar a\|^2$，跨模型报告卡只用 $\tilde D$）。`tests/test_wma_step1.py` 7 项全绿（TDD，G1a–c 合成认证门即单测）。
- **G1 合成认证（PASS 3/3）**：G1a 完美预测器 $|G_{T\to P}|{=}1.5\mathrm{e}{-4}$ / $L_{\mathrm{sym}}{=}0.086$ / $\tilde D_{T,T}{=}0.018$；G1b 破坏预测器 $\tilde D_{P,P}{=}1.07$、$D_{T,P}/D_{T,T}{=}184$；G1c 捷径注入 $L_{\mathrm{sym}}{=}36.4$（G1a 的 425×）、$D_{P,P}{=}0.021$ vs $D_{P,T}{=}2.28$——AITS-P 病理签名按构造复现。
- **G3 infra parity（PASS）**：step91 G0b 一步误差逐位复现（0.16451/0.28946，diff $<10^{-6}$），scale01 选择一致——loader/编码底座与 paper2 管线同源。
- **数据**：swm/PushT-v1 + WeakPolicy 单 env 复刻（dist_constraint=100，逐集 seeded），132 集 × 40 模型步 = 5016 转移；episode 级 80/20 train/test（val 12.5% 在 train 内按转移切，非 episode 级——披露）。动作对齐源码级验证（swm `buffer._gather_clip`：末行 chunk 驱动 $F_t\to F_{t+1}$）。
- **G2 对拍（FAIL → 预注册 fallback 执行）**：$\tilde D_{T,T}{=}0.936$ / $\tilde D_{P,P}{=}0.462$ / $\tilde D_{P,T}{=}3.65$；$|G_{T\to P}|{=}0.686>0.5$、$L_{\mathrm{sym}}{=}6.90>2.0$。与 ATM Table 6 同模型同环境行（$D_{T,T}{=}0.183$ raw、$L_{\mathrm{sym}}{=}0.226$，判健康）**方向相反**。fallback 排查：动作归一化同约定（$[-1,1]$ 原生，$\mathrm{Var}(a){=}2.42$）；剩余自由度 = 探针数据分布（论文未披露其行为数据；WeakPolicy 块内动作弱相关 → 边界帧可恢复性低）。
- **容量披露**：病理形态跨全部三个宽度稳定（$D_{T,T}$ 2.26–2.33、$D_{P,P}$ 1.12–1.13 平；非对角随容量恶化）——非探针容量伪影。注册的 mini-scan 首跑即兑现价值。
- **post-hoc 控制（wma-step1b，明确标注：诊断，非预注册门）**：
  - A 对齐 A/B：registered/shifted/zero = 0.1657/0.1657/0.1648（spread 0.55%）→ INSENSITIVE——一步误差在范数层面对动作条件数值盲；对齐正确性由源码级验证承担。
  - B 动作烙印置换：喂 roll(1) 置换的 $a'$ → $\tilde D_{T,T}(a'){=}1.001$（sanity：真实转移与 $a'$ 零互信息）、$\tilde D_{P,P}(a'){=}0.435 \approx$ 真动作的 0.462 → **IMPRINT CONFIRMED**：预测域动作码是条件化回声（低范数但线性可解码子空间），与环境真实无关。
- **发现（工具文素材）**：
  - **F1（机制）**：动作条件 predictor 把喂入动作烙印进 $\hat z$；ATM 的 $D_{P,P}$ 在此类模型上测到的是回声而非接地动力学——instrument II 必须配烙印置换控制（本步新协议件，ATM 原文没有）。与 PoR 的动作条件化悖论（externalization，arXiv:2606.07687 Table 6）同构互证。
  - **F2（方法论）**：ATM 判定是 (checkpoint × 探针分布) 的联合属性，非 checkpoint 单独属性——同一官方 LeWM/PushT，论文分布判健康、WeakPolicy 分布判病理。报告卡必须钉死探针分布。与 paper2 step99 教训（测量只在 regime 内定价）同构。
  - **F3（交叉验证面板首例）**：LeWM/PushT 三仪器并排——谱 $\lambda_1\approx 0$ ABSTAIN（step91）/ 部署 bias-abstain 勿部署（step97）/ ATM 动作语义自指（本步）——互相看不见的失效面，面板论点的第一张实数据卡。
- **红线自查**：不宣称"LeWM 规划已坏"（烙印子空间低范数，对 CEM 距离代价的影响未测）；$\tilde D_{T,T}$ 高含环境信息地板成分（WeakPolicy 块内动作从边界帧本就难全恢复）——稳健结论是**域不对称 + 烙印**，不是"表征差"。
- **产物**：`papers/figures/wma_step1_atm_lewm.json`、`wma_step1b_controls.json`；`experiments/wma_step1_atm_probe.py`、`wma_step1b_controls.py`；`src/audit/atm.py`；`tests/test_wma_step1.py`。
- **下一刀**：wma-step2——ATM+烙印控制列扫 Tier 0（TD-MPC2 任务格、V-JEPA 2-AC×DROID），矛尖问题：**非动作条件 vs 动作条件架构的烙印对比**（TD-MPC2 的 dynamics 也是动作条件——烙印是否普遍？）。

## [2026-06-11] wma-step1c | 回头看：F2 超claim 更正 + 三项直测全绿

- **动机（自查嫌疑清单）**：① F2 写了"同一官方 LeWM"——但 ATM Table 6 的被诊断模型是其**自训 LeWM-style 受控变体**（训 AITS 变体所必需），官方 ckpt 从未被确立，差异归因横跨 {ckpt 来源 × 探针分布 × 实现细节}；② step1b 烙印控制的 roll(1) 置换在集内有序数据上 $a'$ 多数 = 同集 $c_{t-1}$，"环境无关"措辞不准（结论当时靠 sanity cell 撑住）；③ 运行间可复现性未验证（paper3 同日立案 e2cnn-on-MPS 同 seed 不可复现的教训：验证而非假设）。
- **(i) 可复现性直测：PASS，rel diff = 0.00e+00**——端到端重跑（重采集 + 重训探针）width-256 lane 与记录值完全一致。CPU f64 编码 + CPU f32 seeded 探针栈确定性**验证成立**（栈级对照：paper3 的 MPS 不可复现问题不在本管线）。
- **(ii) 分布敏感性第一手直测（F2 修正后的新立论）**：同一官方 ckpt、同管线、仅换注册行为策略——WeakPolicy：$\tilde D_{T,T}{=}0.936$ / $L_{\mathrm{sym}}{=}6.90$ vs uniform-random：$\tilde D_{T,T}{=}1.002$ / $L_{\mathrm{sym}}{=}5.00$。**幅值显著移动 → 分布依赖第一手成立（不再依赖论文对比）；烙印签名跨分布稳健**（$\tilde D_{P,P}$ 0.462 vs 0.478）。random 下 $\tilde D_{T,T}{=}1.002$ = 信息地板全满（PushT 边界帧对 iid chunk 零可恢复）——反证 ATM 原文 $D_{T,T}{=}0.18$ 来自非常不同的数据 regime 或约定，**其代码未释出，不可单一归因**。**F2 最终版**：ATM 幅值是 (ckpt × 探针分布 × 实现细节) 的联合属性；本步 step1 条目中的"同一官方 LeWM 判健康/判病理"对比表述以本条为准作废。
- **(iii) derangement 烙印：CONFIRMED 无星号**——跨集占比 99.36%，$\tilde D_{T,T}(a'){=}1.005$（sanity）、$\tilde D_{P,P}(a'){=}0.445$（≈ roll(1) 的 0.435 ≈ 真动作 0.462）。**F1 升强：烙印对置换方案与行为分布双稳健**。
- **G3 措辞复核**：实际 diff **恰好 0.0**（强于记录的"<1e-6"）——符合 repo 等式门标准，措辞成立。
- **产物**：`experiments/wma_step1c_lookback.py`、`papers/figures/wma_step1c_lookback.json`。
- **更正传播**：快照页 P4-D 小节已同步改写；vault log 追加更正条目（仅追加文化，旧条不改）。
