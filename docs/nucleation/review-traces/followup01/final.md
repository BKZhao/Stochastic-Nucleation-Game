# N03完成后独立裁定

用途：保存结果审查痕迹。Owner：研究执行者；reviewer：`/root/review_followup`。
输入：冻结配置/协议、confirmation.json和512个原始完成任务；输出：原始裁定与路由。
相关：[主张裁定](../../claim_verdict.md)、[结果](../../followup_results.md)。

## 提示

请只读核对完整512条N03报告、配置与原始任务，给出claim_supported及适用范围并识别阻塞风险。
报告为random149/256、compact3/256、delta=.5703125、Newcombe95[.505227,.629588]、Fisher p=1.53736e-53。
我们只主张登记参数下两个具体生成协议的风险差，H3仍不支持。禁止修改文件或运行生产模拟。

## Reviewer原始回复

claim_supported：yes，仅针对预登记的两种几何生成协议比较。confidence：high。未发现阻塞风险。H3仍不支持。

- random：149/256，风险58.20%。
- compact：3/256，风险1.17%。
- 风险差：57.03个百分点，Newcombe95%区间[50.52，62.96]个百分点。
- 预登记双侧Fisher：p=1.53736×10⁻⁵³。

what_results_support：

在L=48,b=1.16,D_r=.40,μ=0,m=16,T=10000及其余固定参数、共同全合作/全资源初态下，均匀无放回随机选取损伤节点的协议，比所测试的紧凑斑块协议产生更高的持续宏观耗竭概率。两组实际策略损伤和资源损失相同。

what_results_dont_support：

不能推广为所有分散布局比集中布局危险；不能单独归因为周长、连通分量或间距；不证明自发成核、单核稀有逃逸、临界核或永久灭绝。未耗竭未恢复者仍是未决生物结局，不能称已恢复。

独立验证证据：

- 重新调用只读分析函数，校验全部512个任务的COMPLETE、事件/删失时钟、完整样本量和实际初态等成本；主要报告字段全部匹配。
- 从512个events.npz独立汇总，得到149与3个事件。
- 按各自seed重建全部512张几何掩膜，与保存初态逐一一致。
- 512个seed唯一，与已有N01/N02样本无重叠；来源均为1c1e597 / de69fecfbc13f531，working_tree_dirty=false。
- 用另一种有理数超几何表达式独立重算Fisher，结果一致。
- 恢复/未决计数一致：random为8/99，compact为134/119。

证据文件：configs/nucleation/n03_geometry_confirmation.json、docs/nucleation/confirmation_plan.md、data/nucleation/analysis/cc62c28283b1d96a/confirmation.json及其512个原始任务。

missing_evidence / next_experiments：

这个窄主张不需要追加样本才能成立。若要扩展机制或一般性，需要独立预算/参数/布局验证及空间路径分析。H3继续单独调查，不能由本次H1结果替代。

未重新执行物理轨迹，未从稀疏标量重建逐步确认事件；该部分依赖已测试的在线事件检测器。未修改文件、未运行生产模拟。未发现论文级PAPER_CLAIM_AUDIT，因此本裁定确认结果支持范围，不代表整篇论文已完成审计。

## 解析与路由

- scoped_H1: yes；H3: no；confidence: high；paper_claim_audit: unavailable。
- 路由：保留此局部确认，未来研究转向适用范围与机制；H3保持补证，不再把该点长确认寿命直接解释为成核等待。
- 新增绘图入口后最终分析目录为71b13393bb500003；主agent核对两个报告除analysis_sources外完全一致。图形代码变化没有改变输入或统计结果。
