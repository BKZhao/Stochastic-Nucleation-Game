# N03：固定损伤预算的一个独立确认比较

用途：检验选定参数下随机散点与紧凑斑块的耗竭风险差。
Owner：研究执行者。输入：[第二轮计划](followup_plan.md)及其完成清单/诊断；输出：独立256+256副本、风险区间和一个预先指定检验。
范围：L48，b1.16，D_r=.40，μ0，m16，T10000；不外推到其他预算、参数或自发成核。

## 进入依据与冻结协议

第二轮探索在相同节点损伤与资源损失m16下，random为8/16、compact为0/16。
这一结果来自多格探索，只负责选择比较；不得与确认批次混作独立证据。
在N03任何新轨迹启动前固定本文件与 `configs/nucleation/n03_geometry_confirmation.json`。

- 唯一主比较：random减compact的确认耗竭概率差。双侧检验，未预设方向。
- 每组256个全新动态seed；几何生成仍使用独立Python RNG，每个random副本均匀无放回抽取16节点，允许偶然相邻；不是固定某一随机布局的重复测量。
- 初态全合作/全资源；两组均设置恰好16个节点s=0,r=0。中心按相同协议随机周期平移。
- 主终点仍为rho≤.2且r≤.2连续250步的首次确认；确认后停止，否则观察到10000步。主终点不改变。
- 主报告：各组事件数、Wilson95%区间、概率差点估计、Newcombe-Wilson无连续校正95%区间及双侧Fisher精确检验p值。两侧定义为在固定边际下概率不高于观察表的所有可行表之和。
- 唯一预先指定主检验alpha=.05，无中途显著性查看/提前停止；512个任务全部结束后分析。无新增多重主比较。
- 耗竭时间、恢复与未恢复为次要描述，不将它们替换为显著主终点。
- 支持条件：完整任务与等成本检查通过，独立批次Fisher双侧p<.05且风险差为正时支持random风险更高；若显著为负则如实报告反方向；无支持则保留阴性结果，不改终点或追加到显著为止。

这只确认该参数、损伤预算和观察窗下的形状协议效应；不估计m50，不建立唯一临界核，也不证明资源环带机制。
N03复用现有单副本runner，仅允许新的实验ID/入口；不修改物理内核。

方法来源：R stats官方fisher.test文档（https://stat.ethz.ch/R-manual/R-devel/library/stats/html/fisher.test.html）。
风险差区间参照statsmodels官方实现的newcomb方法（https://www.statsmodels.org/stable/_modules/statsmodels/stats/proportion.html#confint_proportions_2indep）。
Fisher检验与Wilson区间不要求额外依赖，使用标准库组合数枚举2×2固定边际分布；合成表验证。

冻结campaign hash：`ece1d6b170fb3c1c`。分析入口拒绝任何偏离该配置的输入；改变设计必须另行登记，不能沿用当前确认主张。
