# paper2 投稿周清单(2026 年 9 月,ICLR 2027 截稿前)

> 状态:稿件已封版(2026-06-10,联合通读 R1–R8 完成,commit `6660b17`)。
> 本清单 = 投稿周的全部机械动作。科学与文本不再动,除非 lit sweep 出现新撞车。

## Claude 的机械动作(~半天)
- [ ] **箱子回线后欠账**:重建三份 PDF(`python3 papers/iclr_submission/build_iclr.py && make iclr-build && make paper2-build`),
      复核 References 起始页(≤ p10 顶 = 主文 ≤9 页),提交 PDF。← 这条不必等 9 月,箱子一开机就做
- [ ] **ICLR 2027 kit 替换**:官方 kit 放出后,`build_iclr.py` 里 `iclr2026` → `iclr2027`(常量 + ensure_template URL
      + preamble 的 \usepackage 名),重建并重验 9 页线(2027 若改页限,以新规则为准)
- [ ] **lit sweep 终扫刷新**:重跑查新(四个 novelty 主张 + E16 的 "to our knowledge new"),附录 E 日期改为扫描日;
      若撞车 → 只动 related work/附录 E,正文主张降级需用户批准
- [ ] **匿名化 artifact 重打包**:`scripts/make_anon_artifact.py`(零泄漏自检必须过;新文件 step94–99/tests 已入扫描范围确认)
- [ ] bib 编译核:natbib 渲染无 `?`;BRo/UWM 两个内联 arXiv id 决定是否升级为正式 bib 条目
- [ ] 全套测试最后一跑(当前 226)

## 用户的按钮(只能你按)
- [ ] OpenReview 注册/共同作者信息
- [ ] 最终冷读一遍 9 页 main.pdf(尤其 step65 图 72% 宽度的纸面可读性)
- [ ] 提交

## 已知冻结决策(别再争论)
- E2 两图在附录 A,step65 留正文(2026-06-10 用户签)
- hero 图 60/62%(p3 浮动页双稳态,阈值在 62/64 之间——别手痒调大)
- "real-robot data, offline monitoring" 措辞红线
- 四个 INCONCLUSIVE + G1b/G8-E 的 as-registered 失败照登
