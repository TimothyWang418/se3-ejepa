# wma-step2 seed：烙印跨架构对比——TD-MPC2 任务格（预注册）

- **日期**：2026-06-11　**所属**：P4-D / WMA　**上游**：wma-step1/1b/1c（台账 `papers/wma_record.md`）
- **矛尖问题**：wma-step1 在 LeWM（teacher-forced ViT predictor）上发现纯回声（$\rho_{\mathrm{imp}} \approx 1.03$）。**回声是动作条件 predictor 的通病，还是训练目标的产物？** TD-MPC2 的 dynamics 同为动作条件 MLP，但以 EMA 编码的**真实** $z_{t+1}$ 为一致性回归目标且受 reward/value 接地——机制上应更接地。
- **scope guard**：3 任务 × seed-1（其余 12 cell 留扩展）；不做 V-JEPA 2-AC×DROID（step3）；不做谱审计重跑（step89 JSON 交叉引用即可）。

## 协议（冻结）

1. **模型/loader**：`models/tdmpc2/{task}-1.pt`，tasks = {walker-walk, cheetah-run, finger-spin}；复用 step89 的 `load_tdmpc2_slices`（faithful 复刻，strict load）。
2. **转移**：1 模型步 = action_repeat 2（step89 同约定）；$z_t = \mathrm{enc}(o_t)$、$z_{t+1} = \mathrm{enc}(o_{t+1})$、$\hat z_{t+1} = \mathrm{next}(z_t, a_t)$；探针目标 = $a_t \in [-1,1]^A$（单动作，无 chunk 歧义）。
3. **行为双 lane（step1c 教训制度化：分布敏感性默认双测）**：
   - lane-π：$a = \tanh(\mu_\pi(z) + 0.3\,\xi)$，$\xi \sim \mathcal N(0, I)$（近专家 + 探索噪声，近似 replay 分布）
   - lane-U：$a \sim U[-1,1]^A$（iid 随机）
   - 各 10 集 × 500 模型步 = 5000 转移；episode seeds $2000{+}i$；episode 级 80/20 train/test，val 12.5% 转移级（step1 同披露）。
4. **ATM**：real 审计全注册协议（widths $\{64,256,1024\}$ × probe seeds $\{0,1,2\}$，主宽 256）；**烙印审计用主宽 256 × 3 seeds**（成本减半，注册为协议而非便宜行事）。
5. **烙印控制（step1b/1c 升格为默认仪器件）**：跨集 derangement（seed 2026，不动点修复，报告跨集占比）；只换被解码的动作（此处即唯一动作）。
6. **读出定义（v1.1 预数据修正案，见文末）**：
   - **回声盈余（主判别量）**：$\eta = \tilde D_{T,T} - \tilde D_{P,P}$——预测域动作信息**超出**真实转移所含的部分。完美反事实模型 $\eta \approx 0$（其 $\hat z'$ 对 $a'$ 可解码是**胜任**不是病理）；LeWM step1 参考值 $\eta = 0.474$。
   - **烙印比（归因读出，降级为辅助）**：$\rho_{\mathrm{imp}} = \frac{1 - \tilde D_{P,P}(a')}{1 - \tilde D_{P,P}(a)}$，仅当 $\tilde D_{P,P}(a) < 0.9$ 时定义，否则 NO-SIGNAL。**注意 $\rho_{\mathrm{imp}}$ 单独不可判病理**——完美模型与纯回声模型都 $\approx 1$（前者解码的是正确反事实、后者是复读）；它只回答"P 域动作信号是否跟随任意喂入动作"。病理签名 = $\eta > 0$ **且** $\rho_{\mathrm{imp}} \approx 1$。

## 预注册 gates

| Gate | 判据 | 备注 |
|------|------|------|
| **G-s2a-η（盈余门，带管辖权声明）** | $\eta < 0.2$ 于 ≥2/3 任务（lane-π）——**附带 $\tilde D_{T,T}$ 天花板披露**：若 $\tilde D_{T,T} < 0.2$（动作显形体制）则 $\eta$ 门在该格**空洞**，如实标 VACUOUS-BY-CEILING 不计入通过 | $\eta \le \tilde D_{T,T}$ 恒成立——盈余只在环境藏动作的体制可测（v1.2） |
| **G-s2a-sym（码相容门，非空洞主张）** | $L_{\mathrm{sym}} < 2.0$ 于 ≥2/3 任务（lane-π；step1 健康/病理分界同款阈值） | TD-MPC2 一致性损失回归真 $z_{t+1}$ ⇒ T/P 码应相容（与 G1c 捷径码相对）。**FAIL 同样可发表** |
| **G-s2b（sanity，每格）** | $\tilde D_{T,T}(a') > 0.9$ | 置换动作不得从真实转移解出 |
| **G-s2c（infra）** | 一步预测 rel-err 中位数 $< 1$；step89 谱列可交叉引用 | loader/语义 sanity |

**fallback（预注册）**：若 lane-π 下 $\tilde D_{P,P}(a)$ 也 $\approx 1$（连真动作都解不出 ⇒ NO-SIGNAL 全格）→ 检查动作-潜变量耦合强度（$\|\partial\,\mathrm{next}/\partial a\|$ JVP 探针），区分"无回声"与"动作通道弱"——两者部署含义不同，如实分报。

## 产出

- `src/audit/atm.py` 新增 `imprint_ratio()` + `derangement()`（TDD，G1 风格合成单测）
- `experiments/wma_step2_tdmpc2_atm.py` + `papers/figures/wma_step2_tdmpc2_atm.json` + 台账条目

## 红线

容量披露随结果走（real lane 三宽度）；不重训任何模型；step89/91 文件只读；不碰 paper3 领地。

## 修正案记录

- **v1.1（2026-06-11，预数据——任何采集/审计运行之前）**：gate 量从 $\rho_{\mathrm{imp}}$ 改为回声盈余 $\eta = \tilde D_{T,T} - \tilde D_{P,P}$。原因：注册后立即的判别力预演发现完美反事实模型在置换控制下同样 $\rho_{\mathrm{imp}} \approx 1$（解码正确反事实 ≠ 复读），$\rho$ 单独无判别力；$\eta$ 才区分"胜任"与"盈余"。此修正同时把 step1 LeWM 结论的承重点显式化：当时钉死病理靠的是域不对称（$\eta = 0.474$）+ 置换归因双证据，仅置换不充分。判别力预演将做成合成单测（完美模型 $\eta \approx 0$ 但 $\rho \approx 1$；回声模型 $\eta$ 大）。
- **v1.2（2026-06-11，预数据）**：$\eta$ 自身的管辖权修正——$\eta \le \tilde D_{T,T}$ 恒成立，故在动作显形体制（DMC proprio 预期 $\tilde D_{T,T}$ 低）$\eta$ 门**结构性空洞**。处置：G-s2a 拆为 η 门（带 VACUOUS-BY-CEILING 标记规则）+ $L_{\mathrm{sym}}$ 码相容门（该体制的非空洞主张）。读出分类学定型为三轴 $(\eta, L_{\mathrm{sym}}, \rho_{\mathrm{imp}})$ + 体制天花板 $\tilde D_{T,T}$——"审计读出有管辖权"（paper2 主题在仪器层的复现，工具文素材）。合成判别测试相应加第三臂（B=0 藏动作世界 + 回声注入 ⇒ $\eta$ 大）。
