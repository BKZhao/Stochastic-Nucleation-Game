# 单节点建立机会：有限判别协议

用途：检验现有几何差异能否由一个最简单的独立通道基线解释。
范围：N02机制筛查，不是第二次N03确认，不判定成核或亚稳性。
Owner：研究执行者；输入：[N03确认](followup_results.md)、原模型及固定配置；输出：256条独立单节点轨迹与描述性基线比较。

## 问题与运行前固定内容

N03 random16风险149/256，尚无单节点风险估计。compact16不能替代单节点。
若每个种子都以相同概率P1(T)在完整L48系统中于T前触发原B终点，且把16个通道视为独立、互不改变传播时间，则有

`P16_OR(T) = 1 - (1 - P1(T))^16`。

这是明确的有限时域参照模型，不是实际同域多种子过程的恒等式。
由已知random16风险反推所需P1约5.31%，只用于设计判断；不把它当实测P1。

- 配置：`configs/nucleation/n02_single_founder.json`，master_seed=202610064，固定256副本。
- 全合作/全资源初态，随机平移的单节点设为D/r0；其余模型、L48、b1.16、D=.40、μ0、T10000与N03一致。
- B仍为rho≤.2且平均资源≤.2连续250步的首次确认；不改变终点以适应结果。
- 使用已有runner；每个job一次播种、连续推进；不更改物理内核。
- 完整预算后统一分析；无中途显著性检验，不依据本轮结果继续追加同批样本。
- 检查256个seed彼此唯一且不与既有正式配置重叠，核验初态、COMPLETE、事件/删失时钟。
- 源码、配置和测试进入运行时指纹与源码压缩包；工作区未提交状态如实写入provenance。不以Git提交号单独代替实际源码版本。

## 分析与可解释范围

分别报告P1、N03 random16、compact16的事件数、Wilson95%区间、恢复和未决数量。
将P1及其区间经单调函数`1-(1-p)^16`变换，另列N03风险所需的等效P1及其区间。
两个95%区间的重叠或不重叠不充当alpha=.05检验；没有预设显著性裁定，也不检验等效。

若基线远低于观测风险，应继续分辨传播时间缩短、多个局部群体相互作用、资源场重叠和合并；不能直接称“集体成核”。
若相近，独立机会解释仍可行，应优先检验跨m外推；不能据此宣称通道独立。
若基线远高于观测风险，应检查干扰/竞争、有限区域饱和及单种子增长时间的可迁移性。
单节点发生事件，只证明该外部种子在有限观察窗内能触发B；不证明自然突变能够产生同一状态。

## 验证与预算

`PYTHONPATH=code:experiments python -m unittest discover -s tests -v`

`PYTHONPATH=code:experiments python experiments/run_nucleation_seeds.py --config configs/nucleation/n02_single_founder.json --workers 8`

`PYTHONPATH=code:experiments python experiments/analyze_nucleation_founders.py --single configs/nucleation/n02_single_founder.json --comparison configs/nucleation/n03_geometry_confirmation.json`

上限5.90e9 site visits；参照N03每副本约2.7秒的既有记录估算数分钟内，实际时间单独记录。
保留全部原始数据和失败记录；不启动新参数大扫描、FFS或WE。
独立只读reviewer已在运行前认可本有限筛查及其解释边界；完整痕迹见后续可行性裁定。
