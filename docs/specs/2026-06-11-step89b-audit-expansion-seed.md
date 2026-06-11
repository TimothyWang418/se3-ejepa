# step89b — E13 审计扩张(15 → 全公开单任务 TD-MPC2 zoo),预注册

**宇宙(冻结)**:官方 HF 仓库 dmcontrol/ 下 2026-06-11 枚举的 34 个未审计任务 × seeds 1–3。
**纳入规则(预注册且自执行)**:任务纳入 ⟺ 库存 dm_control `suite.load` 接受其 (domain, task) 映射
(映射规则:首词为 domain,cup→ball_in_cup 特例;余词下划线连接)。TD-MPC2 自定义变体(cheetah-jump、
reacher-three-* 等)**被规则排除而非被挑选排除**——实测交叉列(E16 证明其承重)需要真环境。排除清单写入产物。
**协议**:逐字 step89(同 certify/measure/z0=中点惯例/eps={0.05,0.1,0.2});维度运行时推导(action_dim 取
env spec,obs_dim 取 reset 展平观测;strict=True 加载即隐式断言编码器维度)。obs 展平 = spec 序,与原五任务
同约定。**增量可续跑**:每 cell 即写 JSON,已有键跳过。
**诚实曝险(先声明)**:新 cells 落入分类学哪格未知;比例怎么变,E13 叙述就怎么如实改写。
