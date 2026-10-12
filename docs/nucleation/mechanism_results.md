# 机制轮结果：固定总量/分布/均值下的局部分配不改变命运（H-E′）

用途：报告 [预登记](../../data/nucleation/N06/campaigns/preregistered_mechanism_predictions.json)（registered 2026-10-07T05:31:53Z，先于任何续跑）的机制轮检验结果；允许被否定。
Owner：研究执行者。输入：冻结预登记、机制命运 campaign（3,840 条 T40000）、剂量伴随（3,840 条 T20）、coupling-0 负对照（48 母态 × 5 臂 × 500 sweep）。
日期：2026-10-07。分析：`data/nucleation/analysis/e2c19f4679d31c60/mechanism.json`。

## 设计

对方向轮同一批 48 个紧凑母态（compact_16、最小种子规则、身份绑定），固定策略图样与资源**总量、值分布、子步前界面均值**（多重集保持，逐母态校验 dB=0.0000 精确成立），仅改变"哪个界面合作者持有哪个资源值"：

| 臂 | 构造 | 预测（预登记冻结，逐母态精确计算） |
|---|---|---|
| interface_protect | 最大资源值→最大有效权重 W_u（W=(1−D)w+(D/4)Σw_v，子步混合后的目标权重） | dJ_pre_exact −3.47 [−6.88,−1.45]，48/48 为负 |
| interface_harm | 反向配对 | dJ_pre_exact +3.77 [+1.53,+8.86]，48/48 为正 |
| interface_shuffle | 冻结随机置换同一多重集 | dJ_pre_exact +0.50（中间） |
| defector_shuffle | 背叛者节点间置换（D 侧资源不进收益，仅经一步扩散以 D/4 前置因子泄漏进界面） | dJ_pre_exact +0.39（梯度对照） |
| original | 基准 | — |

harm−protect 精确 dJ_pre 对比 **+7.24 [最小 +3.98]，48/48 同号**——预测侧的对比度极大且方向一致。

## 裁定

| 预登记预测 | 结果 | 裁定 |
|---|---|---|
| **主检验**：母态配对 q_B(harm)−q_B(protect)>0，聚类 bootstrap 95% 区间不含零 | **−0.0104 [−0.0521, +0.0312]**，21/48 为负 | **未确认**。点估计与预测方向相反但跨零；界内配对命运的局部分配通道在该对比度（\|dJ\|≈7.2）下检测不到 |
| 次级 (a)：shuffle 幅度 < 主对比一半； (b)：defector 等效边界 | 主对比为零时两判据退化（界 0.005/0.0026 < 全部臂区间） | 主检验否定后不作为独立信息；有效次级证据在剂量链 |
| 次级 (c)：剂量链 dJ_pre→a₁ | **配对 ρ(dJ_pre_exact, Δa₁)=0.782 [0.724, 0.835]；配对斜率 0.670；逐臂 a₁ 校准斜率 0.62–0.72** | **确认**：各臂按预登记精确预测产生首 sweep 漂移踢击 |
| 次级 (c)：Δa₁→Δq_B | **ρ=−0.017 [−0.159, +0.124]** | 踢击**不传递**到命运 |
| 次级 (d)：线性敏感度校准 | 斜率 0.673 [0.618, 0.725]（Spearman 0.941） | 完整重排超出线性域，tanh 饱和使响应衰减约 1.5×；如实报告 |
| 负对照（coupling=0） | 48/48 母态五臂策略轨迹**逐位一致**（2 个母态平凡：退化多重集） | **通过**：臂间差异确经资源→收益通道，非任务身份/数据处理伪象 |

各臂 vs original 的命运差（全部跨零）：protect +0.0221 [−0.0221,+0.0651]、harm +0.0117 [−0.0352,+0.0560]、shuffle +0.0195、defector +0.0065——四臂共同抬高约 +1–2pp 的"任何扰动"共模（噪声量级内）。

## 机制解读：暂态踢击 vs 持久结构（本轮的核心贡献）

1. **踢击真实且被精确预测**：harm/protect 臂的首 sweep 漂移按预登记 dJ_pre 定量实现（斜率 0.67，饱和衰减与线性校准一致）。
2. **踢击在数个 sweep 内衰减**：多重集保持的重排是界面带内的高频扰动；资源子步的扩散（每步衰减因子 1−D(1−cos k)，带内波长 2–4 格点）与局部再生把场拉回**唯一**的策略固定平衡（收缩映射，已有证明）——分配差异在 ~3–5 个 sweep 内被抹除。
3. **冲量估算**：踢击 ×3.5 J 单位 ×0.67 转化 ×~3–5 sweep ≈ ~10 个背叛者，相对宏观阈值可忽略——与零命运效应定量一致。
4. **与方向轮对照**：中幅位移臂的 ΔJ_pre（~10–15）与本轮踢击幅度**同量级**，却产生 ±6pp 命运效应。差别不在幅度而在**持续时间**：位移使资源凹坑相对策略图样持久错位（界面移动前不恢复），ΔJ 持续整个建立期；重排只改带内分配，结构性坐标（B、凹坑几何）不变。
5. **结论**：命运分流由资源场的**慢结构坐标**（凹坑—界面几何、界面均值 B）控制，而非界面的瞬时平衡涨落。这统一解释已有全部证据——瞬时 J 无命运预测力（ρ_S≤0.14）、B 秩相关确认（−0.31）、uniform_live 持久结构破坏 −12.1pp、本轮分配零效应。

## 与既有证据的整合（资源通道分解阶梯）

| 干预 | 固定什么 | 改变什么 | 持续性 | 命运效应 |
|---|---|---|---|---|
| interface_protect/harm（本轮） | 总量、分布、界面均值 | 带内分配 | ~3–5 sweep（扩散+平衡） | **0** [−0.05, +0.03] |
| uniform_live（确认轮） | 总量 | 持久抹平空间结构 | 永久 | −12.1pp [−17.3,−6.7] |
| 中幅位移（方向轮） | 总量、分布 | 持久错位凹坑，ΔB≈+0.4 | 界面移动前持续 | −6.3pp；与 dB 秩相关 −0.31 |

下一分析（数据已存在，无需新轨迹）：以早期 D 轨迹积分（∝∫J dt）为中介变量，跨方向轮+机制轮全部臂检验"冲量而非瞬时漂移"表述。

## 来源、复算与验证

| 批次 | campaign | 运行数 |
|---|---|---:|
| 机制命运 | N06 `7013ea1e7cc81e35` | 3,840 |
| 机制剂量 | N06（`tmp/mechanism_dose_run.log`，config `n06_mechanism_dose.json`） | 3,840 |
| coupling-0 负对照 | `analysis/mechanism_coupling0.json` | 48×5×500 sweep |

- 预登记：`preregistered_mechanism_predictions.json`（先于任何续跑；含逐母态逐臂 dJ_pre 精确/线性、不变量校验、敏感度画像；harm 线性预测对 defector 泄漏 17.5× 通过 3× 门槛）。
- 中途三次源码修正（size 接口等）期间的局部完成目录已隔离于 `.quarantine-*`，最终 3,840 条全部为单一源码版本。
- 103 项测试通过（94 + 机制轮 9 项：权重数值导数、多重集/总量/均值不变量、重排不等式排序、精确 J 排序、泄漏结构、退化、coupling-0 一致性）。

```bash
PYTHONPATH=code:experiments python experiments/preregister_mechanism_round.py
PYTHONPATH=code:experiments python experiments/run_nucleation_fate.py --config configs/nucleation/n06_mechanism_fate.json --workers 48
PYTHONPATH=code:experiments python experiments/run_nucleation_fate.py --config configs/nucleation/n06_mechanism_dose.json --workers 48
PYTHONPATH=code:experiments python experiments/check_mechanism_coupling0.py
PYTHONPATH=code:experiments python experiments/analyze_nucleation_mechanism.py
```
