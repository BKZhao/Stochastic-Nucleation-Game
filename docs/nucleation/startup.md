# 随机成核研究启动包

用途/范围：成核研究接口与模块契约，覆盖N00、N01、N02及单个N03比较；不代表完整研究完成。
输入：[研究设计](research_design.md)及基线接口。
输出：新增代码、测试、JSON 配置、独立逐运行数据及筛选报告。
Owner：研究执行者。相关：[文档入口](../README.md)、[执行记录](results.md)。

## Explore

实际起始 commit：`2d89df0569c05846b7b93f3925cf495bb80162a2`，与设计一致。
工作区 `eco/nucleation`，分支 `research/nucleation-start`；`eco/upstream` 保留原仓库。
设计引用的 `03_现有仓库接入与增量开发清单.md` 未随文件提供，使用第 12 节接口契约。
原仓库无测试框架配置、环境锁文件或 formatter/linter；使用标准库 unittest，记录现有环境。

## Design / Implement

| Owner | 输入 | 输出/依赖 | 边界 |
|---|---|---|---|
| research_events | 每步联合条件 | 起点、确认时间、计数器 | 不修改旧事件函数 |
| research_clusters | 周期布尔格 | 标签、面积、周长、拓扑绕行 | 不使用触边代替绕行 |
| research_state | 完整 SimulationState、上下文 | 无别名副本、原 dtype 快照 | 不声称复制 RNG |
| nucleation_seeds | L、m、形状、独立生成器 | 精确 m 节点掩码 | 全资源背景等成本 |
| nucleation_survival | 事件/行政删失表 | KM、RMST、Wilson | 技术失败不纳入生存样本 |
| research_common | 显式研究根、配置、运行 ID | 原子写入、校验后 COMPLETE | 拒绝旧路径和完成任务覆盖 |
| nucleation_progress | 连续sweep、D总数；完整μ0队列 | 首跨时钟；依赖nucleation_survival的历史风险 | 只读观测、不消费RNG；人口阈值不是核标签 |
| nucleation_runner | 冻结配置、单个 seed | 完整单副本结果、阈值快照；依赖progress | 连续step_state；resource_value缺省0，资源干预仅改变植入初态 |
| nucleation_campaign | JSON矩阵、master seed | 独立配置与seed | 整体damage轴与damage子轴互斥，保持旧展开结果 |
| run_nucleation_* | JSON 矩阵 | 进程隔离的 job、汇总 | 不交错推进副本 |
| nucleation_growth_analysis | 已校验轨迹、增长时钟与快照 | 依赖周期簇分析的路径/空间描述 | 明确有效分母，不回填B后微观状态 |
| analyze_nucleation_growth | 冻结campaign及逐运行清单 | 历史风险、资源风险差、带源码身份的growth.json | 复用founders初态校验；不做临界核/等效判定 |
| plot_nucleation_growth | growth.json及其空间快照 | 汇总PNG/PDF、规则选定的场示例 | 按最终结局选例，不称典型路径 |

## Verify / Review

- N00：合成事件、周期跨边连通/绕行、分裂合并、等成本形状、删失、复制及输出测试。
- 同环境同 seed，L=16、2 seeds、100 sweeps：观测开关的微观数组逐位一致；step/advance 对齐。
- N01：先实测速度，再执行 24 参数点 ×16 副本 ×10000 sweeps 候选矩阵。
- N02：代表点、小样本、四种形状、精确损伤预算，未达到统计精度前不宣称几何效应。
- 逐步事件以 t=1 为首项；单独存 t=0。全程无准备筛除，time_origin 为 initial_state 或 post_seed。
- 空间采样间隔明确记录；采样簇谱系不等于逐步成功核识别，不从粗采样直接标注单核机制。
- 突变尝试与同访问保留缺陷计数需要内核级观测；本启动包不改内核，字段明确标为未测量，不用 Nμ 替代。
- N03已完成单个冻结独立比较；N04–N08未启动，N06恢复核心尚未校准，未计算committor。
- 长窗/资源观测通过真实后端末态一致、逐步跨越、历史队列和初态资源测试；结果与验证范围见[长窗结果](growth_results.md)。

## Cleanup

保留测试记录、预实验原始数据、配置、执行环境及源文件指纹；临时记录仅在 tmp/。
技术中断任务保留 partial 目录，捕获到异常时另写 FAILED.json，重跑从初态与 seed 开始；未完成任务不进入统计。
