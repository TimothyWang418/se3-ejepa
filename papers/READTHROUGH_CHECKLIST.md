# 作者通读清单 — ICLR 稿(每条 = 你签字的主张 → 证据指针 → 该盯什么)

> 用法:对照 `iclr_certified_horizons.md`(投稿短版)逐节读;每条勾 ☑ 或标 ✗+一句话,丢回给 Claude 修。
> 长版 `paper2_certified_world_models.md` 同构(§5.x ↔ E-x),读一份即可。

## Abstract(4 个钩子,每个都是 reviewer 引用你的那句话)

- [ ] **"Scale buys interpolation; structure buys a certified horizon"** — 全文母题,你最认这句吗?
- [ ] **双侧紧 + 排他**(Thm B + Prop 6 匹配下界;Lemma 2 converse)→ 证据:appendix B,step65 数值 1e-14
- [ ] **可行动 a-priori**(8–16% vs 61–65%,Prop 9 预算定律,重标定要花标定集)→ step85/85c/88 JSONs
- [ ] **审计真模型 + "scale does not buy a calibrated horizon"** → step89/91/92 JSONs
- ⚠ 盯:abstract 现在信息密度极高——读着喘得过气吗?可再删一从句。

## §1 贡献清单(5 条,每条末尾的限定词是防线,别让修改时丢了)

- [ ] (i) 双侧 horizon + 范围定理(Prop 7)+ 连续性(Prop 8)——"learned-model lift rigorous"
- [ ] (ii) 排他性 = **架构不可能性**("proved, not asserted")——这是最强的话,确认证明你读过没毛病(Lemma 2 + freeness 假设)
- [ ] (iii) Noether hinge(守恒 ⇒ 无界 horizon)
- [ ] (iv) Theorem B′ 字面证书(均匀双曲紧、其余自弃权)
- [ ] (v) actionable a-priori + E12/Prop 9 —— 限定词"with zero calibration data"在不在
- [ ] hero figure:三幕(忠实/有价/真实)你看着顺吗?(c) 面板文字略挤,要不要我再排?

## §3 定理区(数学是你的主场,重点核我们后加的两个)

- [ ] **Prop 9**(预算定律):证明 4 行,核 $V(c)=\max(0,L-BH/c-H)/L$ 的推导和 $c\ge1$ 限定
- [ ] **Prop 10**(有限样本区间):核假设链(紧致→有界 log-stretch;平稳+混合→Bernstein)和诚实注记("assumed, not certified");$n\asymp\sigma_\infty^2\log(1/\delta)/\varepsilon^2$
- [ ] §3.4 的 Prop 10 交叉引用句读着自然吗?

## §4 实验区(每条主张 ↔ JSON 可复核)

- [ ] **E2**:R²=0.98 vs <0;相变 N=28→40;GRU 同败(conditional-Lyapunov 论证)→ `step74/83 JSONs`
- [ ] **支撑套件段**(E1,E3–E8 一段速览 + 附录 D)——压缩后你觉得哪条值得捞回主文?
- [ ] **E9**:"2/3 seeds untuned frontier" + 摆环/双摆 class-lift
- [ ] **E10**:cat map 紧(1.00×/1.17×)、Hénon 弃权——"self-diagnoses its regime"
- [ ] **E11**:诚实 INCONCLUSIVE 措辞(你舒服吗?这是亮诚实文化的窗口)
- [ ] **E12**:margins +0.45/+0.50/+0.57;**环的条件因果版**("gap 恰好出现在稠密谱膨胀的 2/5 seeds")——新措辞,读一遍;UQ Pareto 句("唯一 a-priori 点")
- [ ] **E13**:15-cell 范围地图(校准/乐观/弃权)+ LeWM 第二家族 + SimNorm 结构带披露
- [ ] **E14**:"scale does not buy a calibrated horizon"——单 ckpt/cell 的 descriptive 声明在不在;48M 收缩这个反直觉点你信吗(我们只有官方 1 个 ckpt,无法排除训练随机性)
- [ ] **统一段**("audit universal; a-priori guarantee is structure's")——这段是 AC 必问的答案,值得精读

## §5 Related work(novelty 防线)

- [ ] Mo 2605.03338 的区分句(中性模 vs horizon;离散 $\mathbb{Z}_N$ 下空洞)——别过贬,他可能是 reviewer
- [ ] Geng 2512.08991(conformal/统计 vs a-priori/免训练)
- [ ] "to our knowledge new" 出现的每一处(E13 audit、E12 within-method)你都愿签

## §6 Limitations(诚实资产,确认没被稀释)

- [ ] 安全 INCONCLUSIVE(step86)
- [ ] **范围律**(step93 新加):"决策价值集中在被决策量=被认证量处,隔任务映射稀释"——这条是教训也是贡献,语气对吗?
- [ ] 1–2 GPU scope 声明

## 全局三问(合上 PDF 后答)

1. 一句话复述论文,和 abstract 第一句一致吗?
2. 哪个实验你最想砍?(若有 → 说,砍它腾页数)
3. 哪句话你不敢在 rebuttal 里辩护?(那句就是要改的)

—— 读完丢标记给 Claude;修订 → 重编 PDF → arXiv/投稿。
