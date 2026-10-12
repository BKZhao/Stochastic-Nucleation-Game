# 几何与初始资源的有限对照

用途：检验当前几何风险差是否只在同时清空种子资源时出现。
Owner：研究执行者；输入：[可行性路线](viability_assessment.md)、长窗运行及原模型；输出：四格独立风险与空间增长描述。
范围：外部植入D的资源处理，不等同自然突变，不研究准稳态。

## 进入依据与固定设计

用户已接受“延长观察后做资源不清零对照”。长窗192条已完成：40k时single1、compact16、random16分别8/64、9/64、47/64首次确认B。
几何差异在更长窗仍存在，因而继续检查外部资源清零是否为观察到差异的必要条件。
这些旧轨迹不并入本四格比较。

- 固定L48、b1.16、D=.40、μ0及原模型其余参数；背景全C/r1。
- 策略损伤均为16节点；几何compact/random × 种子初始资源r0/r1，共4格，每格64个全新独立seed，合计256条。
- master_seed=202610067，配置`configs/nucleation/n02_resource_control.json`。
- r0将16个种子资源置0，r1保留1；每个资源水平内部两形状的策略/资源损失相同。跨资源水平有意改变资源损失，不称跨水平等成本比较。
- 原B定义、T10000、首次确认后停止；每10步采样、500步完整快照、每sweep增长里程碑与阈值状态同长窗协议。
- 原物理内核不改，只增加损伤接口`resource_value`，缺省0保持旧行为，拒绝非有限值/越界值/布尔/字符串。
- 256条全部完成后统一分析，无中途检验或基于结果加样本。上限5,898,240,000 site visits。

## 分析与分支

主描述为T10000每格事件/恢复/未决、风险Wilson95%区间，以及每个资源水平的random减compact风险差和Newcombe95%区间。
两个几何差的相减仅作交互作用点估计，不由“不显著”或区间重叠宣称资源处理等效。
没有正式主检验或等效检验；64副本预算用于识别大差异，不保证排除小作用。

1000/5000时风险、5000未决队列、增长阈值与空间状态是次要描述，遵守[长窗分析合同](long_horizon_plan.md)：时间队列准确、报告所有分母、D=0的最大簇份额为null、条件时长不叫因果前沿速度。

- 若r1下random仍呈较高风险而compact较低，说明在本外部植入协议中，几何差异也可出现在无初始资源清零时；不能称资源场演化没有作用。
- 若差异主要存在于r0，外部资源耗竭是需要继续解释的条件；不能将r0损伤直接当自然突变。
- 两种结果都不能证明自然成核。sweep边界D/r1会先经历资源更新，而自然突变发生在节点访问时、资源更新之后。

## 验证与记录

测试先复现旧代码会错误清空r1资源，再验证精确初态、配置拒绝及全部回归测试。
核验新seed与此前3008条正式任务不重叠，保存协议副本、SHA-256、实际源码指纹和来源压缩包。
运行后核验256条COMPLETE、初始资源损失、事件时钟及阈值快照。独立reviewer裁定局部结论范围。
数据、图、日志分别进入`data/nucleation/`、`figures/nucleation/`、`tmp/`；不删旧结果，不commit/push。

```bash
PYTHONPATH=code:experiments python -m unittest discover -s tests -v
PYTHONPATH=code:experiments python experiments/run_nucleation_seeds.py --config configs/nucleation/n02_resource_control.json --workers 8
PYTHONPATH=code:experiments python experiments/analyze_nucleation_growth.py --config configs/nucleation/n02_resource_control.json
```
