# 启动执行记录

用途：记录已经执行的验证与预实验，避免把研究设计当作结果。
范围：首轮 N00、N01 与小样本 N02。后续研究见 [第二轮结果与N03确认](followup_results.md)。输入：配置、完成任务及校验清单。输出：结果表与下一步依据。
Owner：研究执行者。相关：[启动契约](startup.md)、[研究设计](research_design.md)、[完整筛选表](screening_tables.md)。

## 代码与验证

物理内核、晶格、旧观测函数及原 `experiments/common.py` 均与基线 `2d89df0` 相同。
N00 的 15 项 unittest 通过，包括两个 seed 的 100-sweep 逐位回归、末步事件、全删失、
周期绕行、精确损伤、子分支播种复跑、完成标记校验与异常配置拒绝。
使用已有 Python 3.12 环境，NumPy 1.26.4、Numba 0.66.0、Matplotlib 3.10.8；逐任务保存实际版本。
没有新增测试框架或修改全局环境。二维 PNG/PDF 已成功生成并人工查看；环境发出 Axes3D 导入警告，本研究图不使用三维轴。

全部 1344 个正式启动任务完成，逐任务文件校验及 clean working tree 来源检查通过，未发现 FAILED.json。
`n02_bracket` 原配置复跑成功校验并跳过全部384个完成任务；未重复模拟或覆盖原始结果。
技术中断与掉电场景没有做系统级故障注入，不能据文件校验声称已经验证断电持久性。

## 已完成批次

正式启动批次使用 `schema_version=1` 的配置。早期性能/开发筛选保留为独立配置哈希，
不与正式批次合并、不作为独立确认样本。所有后续选点均是探索性决策。

| 批次 | 参数格数 × 副本 | 确认耗竭 | 恢复 | 未耗竭未恢复 | 代码 commit | campaign hash |
|---|---|---|---|---|---|---|
| n01 | 24 × 16 | 298 | 0 | 86 | `cb41075` | `b9b63239ce022531` |
| n02_small | 24 × 8 | 192 | 0 | 0 | `cb41075` | `c154a06f7aafbde0` |
| n02_transition | 48 × 8 | 0 | 384 | 0 | `a9b1234` | `a32bb3e9f1c1c921` |
| n02_bracket | 48 × 8 | 189 | 117 | 78 | `8d62f5e` | `9c546fa3fa651b11` |

N01 的“未耗竭未恢复”仅表示未确认耗竭；μ>0 时不评估永久恢复。

## 数据支持的观察

- N01 的 `b=1.12, D_r=0.30, μ=0.001` 在10000步内10/16确认耗竭；Wilson 95%区间为[0.386,0.815]，中位确认时间4757，RMST(10000)=6495.50。它是下一轮局部筛选候选，不是已经确认的亚稳区。
- 5个N01格点0/16事件，其共同观察窗内的单侧95%二项风险上界约0.171；中位寿命未达到，不能称绝对稳定。其余18格均16/16，许多较高b点快速失效。
- N02原代表点 `b=1.20, D_r=0.30, μ=0` 的四种形状、6个预算全部8/8耗竭；结果风险饱和，不能用来证明几何不重要。
- 低b筛选 `b=1.12, D_r∈{0.15,0.30}` 的384条全部恢复，无确认耗竭。此结果与N01有持续突变的协议不同，不能跨协议直接比较风险差。
- 中间值 `b=1.16`：D_r=0.30的192条无确认耗竭，其中115条恢复、77条截至10000步仍未恢复；D_r=0.45的192条中189条耗竭、2条恢复、1条未耗竭未恢复。
- 三个7/8耗竭格点分别为D_r=0.45下的 stripe m=4、compact m=8、compact m=16，其余该扩散值格点均8/8。风险未跨越1/2，也未呈现可靠单调阈值，故不估计m50、不宣称形状效应已证实。

## 进入确认性实验前还缺什么

1. 围绕N01候选点检查活动准备期、条件分布与等待时间分离；同时保留准备期失败。当前所有寿命均从指定初态开始。
2. 针对N02中间值的未恢复轨迹检查簇和资源路径，在D_r=0.30与0.45之间缩小区间；先找到概率过渡，再增加独立样本，不直接启动N03全矩阵。
3. 周期簇与采样重叠关系已保存，但没有逐步谱系和经过committor验证的成功核。回转半径、长宽比、扩散前资源通量、微观突变计数及N06完整分支上下文仍未实施。
4. 现有证据不足以主张随机亚稳、单核/多核交叉、临界核尺度或Avrami规律；N03–N08仍未执行。

## 产物与复跑

数据位于 `data/nucleation/<experiment>/<config_hash>/<run_id>/`；每个run保存初末及间隔状态（float64资源）、标量轨迹、簇、重叠关系、事件和来源。
完成清单含全部文件SHA-256；来源压缩包在 `data/nucleation/_sources/`。原始数据约数百MB，保留本地，不推入Git；完整格点汇总已进入本仓库文档。

源码指纹保守地涵盖 code、experiments、tests 与 configs。添加配置也会改变指纹，旧任务因此拒绝在不同版本下直接跳过；复核旧批次应检出表中的commit或使用该批次来源压缩包。不要删除完成标记绕过检查。

```bash
PYTHONPATH=code:experiments python -m unittest discover -s tests -v
PYTHONPATH=code:experiments python experiments/run_nucleation_seeds.py --config configs/nucleation/n02_transition.json --workers 8
PYTHONPATH=code:experiments python experiments/run_nucleation_seeds.py --config configs/nucleation/n02_bracket.json --workers 8
PYTHONPATH=code:experiments python experiments/plot_nucleation_screening.py --pilot data/nucleation/N01/campaigns/b9b63239ce022531/summary.json --seeds data/nucleation/N02/campaigns/c154a06f7aafbde0/summary.json
```

图：`figures/nucleation/screening.png`、`.pdf`；补充中间值分扩散图位于 `figures/nucleation/bracket-diffusion-*/`。
这些图是预实验筛选图，没有把置信区间或图示趋势当作确认性统计结论。

## Review / Cleanup

只读独立审查已检查配置、事件、RNG、周期绕行、删失与输出，发现的异常配置问题均已修复并有回归测试。
保留清单：所有原始批次、来源压缩包、配置、测试日志、试跑日志及生成报告脚本。
临时记录在忽略的 `tmp/`，原始数据在忽略的 `data/`，图在忽略的 `figures/`；本轮没有删除用户资料或历史产物。
