# step88 ring n=15 机制检验 — 预注册(2026-06-11,新 10 籽仍在跑)

**目的**:把"budget gap 恰出现在 dense λ1 膨胀的 seed"(发表版 2/5 轶事)升级为机制相关检验。

**Flag 定义(由已发表的 seeds 0–4 标定,先于 seeds 5–14 数据冻结)**:
- 膨胀 flag:`lambda1_mlp / lambda1_conv >= 1.5`(旧籽比值 2.05/1.38/2.13/1.16/1.10 — 1.5 干净分离发表叙事的 2/5)
- 赢 flag:`win_margin >= 0.10`(旧籽 0.242/0.035/0.329/0/0 — seed1 的 0.035 不算严格赢,与发表口径一致)

**检验(n=15 合并后)**:
1. 2×2 表(膨胀 × 赢)+ **Fisher 精确检验,单侧**(H1:膨胀 ⇒ 赢)。机制确认线:p < 0.05。
2. 连续版:Spearman ρ(膨胀比, win_margin),报告 ρ 与 p。
3. **反机制格子单独点名**:任何"赢而不膨胀"的 seed 如实列出(它证伪机制的纯净版)。

**诚实条款**:数字怎么落怎么报。若 Fisher 不显著或反机制格非空 → 论文措辞从"the condition is the
mechanism"降级为相应弱化版;gate 纪律同 paper2 全程:阈值此刻冻结,不随结果动。

**产物**:`papers/figures/step88_ring_frontier_n15.json`(5 旧 + 10 新合并,含 2×2/Fisher/Spearman)；
原 5 籽正典文件保留不动(出处完整性)。
