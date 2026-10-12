# 公式、证明与 Lean 映射

用途：说明物理论文推导的前提、依赖和验证边界。Owner：理论模块维护者。
输入：物理内核、界面/重排观测定义；输出：论文推导与 formal/ 验证记录。
完整手工证明的唯一正文：[A_derivations.md](../../paper/drafts/A_derivations.md)；
事件与估计量：[B_checks.md](../../paper/drafts/B_checks.md)。

## Target and status

PROVABLE AS STATED，在下列显式前提下证明局部界面响应、固定策略资源动力学和线性资源分配极值。
精确漂移指首访问，资源收缩比较同一固定策略下的资源场；
有限重排只保证线性目标极值，不保证完整非线性 J 或命运排序。

## Invariant object and assumptions

核心对象是完整微观态下的界面响应，以及固定策略时的资源映射。
长期命运是独立的首次事件实验对象，不由局部 J 代替。

- 周期方格 L≥3、N=L²、四邻居、零突变；首访问均匀点与邻居。
- 资源范围 [0,1]，g、δ、D∈[0,1]；存在唯一性要求 g、δ>0。
- 资源比较使用同一固定策略；K>0；敏感度使用正收益光滑分支。
- 一般有限权重模型：非负且行和 1 负责收缩，列和 1 负责总量守恒，对称性负责 W=Mᵀw=Mw。
- 实数恒等式与浮点/裁剪实现分开；奇裁剪保留互补性，平滑 J 不是裁剪均值。
- 重排集合固定且同为边界 C，局部斜率均为 1−g；有限幅度必须重算真实 Jpre。

## Derivation map and coverage

| 论文对象/标签 | Lean owner 与主要声明 | 仍由正文或代码审阅连接 |
|---|---|---|
| eq:moments, probabilities, drift_bound | Interface: first_visit_mean/second_moment/variance, interface_drift_bound | 均匀有向边抽样及 P≤2N 的格点识别 |
| eq:clipped | Interface: clip_gap_odd, clipped_fermi_complement | clipGap 分段式与实际实现 |
| eq:initial_resource/gap/motifs/isolated | Balance: isolated_defector_bias（代数） | 孤立缺陷四边与 4/3 邻居数，几何 Hac 分组 |
| eq:balance/resource_total/local_balance | Resources: diffusion_preserves_total；Balance: resource_total_balance, covariance_balance | 指示量、格点平均与 N>0 |
| eq:flux | Balance: internal_diffusion_flux_cancels | 固定域内/外边的格点求和识别；不描述移动域 |
| eq:diffusion_mode/mode_bound | Balance: diffusion_mode, diffusion_eigenvalue_bounds, diffusion_mode_lower_bound | 允许周期波数、完整 Fourier 基；不假设非均匀反应下模式独立 |
| eq:contraction/average | Resources: resource_step_bound, decay_bound；Equilibrium: resource_step_lipschitz | 周期扩散权重满足前提 |
| eq:affine_map/equilibrium_linear | Equilibrium: resource_equilibrium_exists_unique, resource_equilibrium_in_cube | 由既有模型整理矩阵形式；逆矩阵几何级数给手工论证 |
| eq:equilibrium_error/returned_error | Equilibrium: resource_equilibrium_error, resource_returned_iterate_error | 实际停止条件是否满足、浮点误差不属于实数证明 |
| eq:sensitivity/edge_derivative | Sensitivity: interface_edge_derivative；Rearrangement: boundary_sensitivity | 边集与正收益分支 |
| eq:chain/effective_weight | Rearrangement: interface_path_derivative, effective_weight_identity | 资源方向变化、格点对称性与 J 边表示 |
| eq:rearrangement/exchange/permutation | Rearrangement: rearrangement_max/min, exchange_gain, permutation_preserves_observable | 代码排序、固定掩膜对应有限置换；不保证非线性极值 |
| eq:autocorrelation/translation_autocorrelation_proof | Alignment: translation_preserves_autocorrelation | 有限周期方格识别为两个循环群；均值由置换不变量保持；除以 N |
| eq:translation_balance/reconstruct_covariance | Observables: translation_resource_balance, covariance_reconstruction | 前者固定策略与初始均值；后者 δ≠g；时刻按资源先更新约定 |
| eq:cumulative_response | Observables: cumulative_response | 对任意实均值序列验证有限伸缩和；有界人数保证期望存在；不建立随机过程语义 |
| eq:partition/reach_decomposition/unresolved_bound | Observables: outcome_count_partition, empirical_reach_decomposition, unresolved_fraction_bounds | 验证有限计数/集合关系；总样本为正、到达集非空，事件包含关系由物理定义连接；不等于总体概率形式语义 |
| eq:absorbing | Resources: copy_constant_strategy, all_c_mean_nondecreasing | 全访问序列和零突变连接 |

eq:clusters、crossing、adaptation、finite_moments、clock、conditional_fate、moment_estimators 是定义。
Observables 对有限计数、集合包含和实数序列/收支恒等式给出 Lean 证明；事件与概率过程仍由正文连接。
二阶 Taylor 余项与 Neumann 逆算符级数有手工推导，不计入 Lean 已覆盖声明。
Taylor 余项采用最大范数、固定有限格点/策略/参数及 K>0，扰动整段保留正收益分支；常数不对 K→0 或增长界面统一。
图示点估计、经验曲线，以及另行记录的 bootstrap 区间和相关性均不是 Lean 定理。

## Verification

学校工作树内运行 formal/README.md 的 lake build 与 Audit.lean。
formal/verification.json 绑定工具链、所有项目 Lean 源码、54 项审计声明和日志校验和。
本轮独立审查、端点核验及证明义务见 [PROOF_AUDIT.json](../../paper/PROOF_AUDIT.json)；未将 Lean 通过扩大为模拟器认证。
Lean/mathlib 标准基础公理允许 propext、Classical.choice、Quot.sound；
不得使用 sorry、admit 或项目自定义公理。
存在唯一性及误差定理验证实际声明的 resourceStep，而非仅给结论作为假设的标量代数。

## Interpretation and non-claims

资源收支中的协方差显示平均量不闭合；它不直接证明任何特定因果机制。
收缩只在同一策略下比较资源场，不能证明耦合系统没有长暂态。
W 是局部子步后场的有效权重；原始场方向导数还必须乘 a_u。
局部排序和短时响应不能替代长期命运实测。
没有临界核、自发成核率、普适类、无限时间相边界或浮点模拟器正确性证明。
