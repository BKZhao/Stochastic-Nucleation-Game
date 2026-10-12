# 第二轮启动前独立审查

用途：实验—主张裁定与代码审查痕迹。Owner：研究执行者；reviewer：独立只读 agent `/root/review_followup`。
范围：首轮证据与第二轮设计/统计实现。相关：[第二轮计划](../../followup_plan.md)。

## 提示与输入

RESULT-TO-CLAIM：审查H1同成本形状风险、H3稳定活动背景与稀有逃逸。
输入为 results.md、research_design.md、screening_tables.md；N01候选10/16、RMST6495.50，
N02在D=.30时0/192、D=.45时189/192。主agent同时读取旧轨迹，报告前2000步全体在险但rho/r下降、D簇扩大。
追加审查 followup_plan.md、nucleation_diagnostics.py、analyze_nucleation_followup.py、runner新增streak与测试。
约束：只读；不提交、不运行生产模拟；可运行测试；报告证据、缺口、风险和验证范围。

## Reviewer原始裁定（结果判断）

claim_supported：no（针对H1、H3当前完整主张）；筛选阶段进展为partial。这表示证据不足，不表示假说已被否定。
confidence：high，针对“尚不能确认H1/H3”的判断；对具体机制判断为low。
paper_claim_audit：unavailable，因此论文主张裁定为provisional。

高风险：条件幸存者偏倚。后期仍存活者的平稳均值可能来自持续筛掉易失败轨迹；τ_B>t还可能包含已经衰退、处于250步确认段内的状态。条件幸存不等于成功建立活动背景。
高风险：事件时钟误释。h_B=250是确认规则；连续区间的t_confirm_B−t_entry_B=249。二者均不是成核等待或传播时间。仅从确认时间减250也不能得到成核等待。
高风险：删失和未恢复混同。N02的77条未恢复轨迹不能按恢复处理，也不能只研究已恢复者。有限时间不耗竭不等于永不耗竭。
中风险：选择与精度。候选来自多格筛选，旧16次应与新64次分开报告；每格8次的区间很宽。重复采样时点不是独立副本。
中风险：时间分辨率。间隔快照与簇重叠只能支持粗粒度路径；不能据此断言全过程只有一个成功核。

## Reviewer原始回复（实现与更新裁定）

新计划可以继续冻结；在本轮固定配置下，未发现高风险统计实现错误。

- landmark_summary正确使用τ_B>landmark；恰在landmark确认者排除，恰在后续5000步截止确认者计为事件。
- 配对窗口使用相同轨迹，排除后窗末端及此前确认者；整条轨迹bootstrap正确。计划明确标注事后幸存者队列，未将无显著漂移解释为平稳。
- 活动门槛同时检查宏观状态、最大背叛簇和持续计数，且明确不是committor恢复核心。
- collapse_streak在确认前记录真实在线计数；本轮全部stop_after_collapse=true，语义成立。

建议冻结前补两项：

1. 中风险：数据完整性的语义检查。read_trace校验采样覆盖executed_sweeps，但将配置Tmax当作行政随访。需拒绝“无事件但提前结束”的记录，并核对event_observed、t_confirm_B、observed_time、executed_sweeps一致，防止完整校验和掩盖不完整随访。
2. 中风险：分析来源指纹不完整。analysis_sources应包含nucleation_survival.py、nucleation_campaign.py、research_common.py。目前这些依赖改变时，可能仍写入相同分析目录。

边界说明：若未来使用stop_after_collapse=false，现有detector在确认后冻结streak，因此它不再表示“当前持续段”；本轮不受影响。

科学裁定更新：旧候选前2000步全部16条仍在险，其合作/资源下降与最大背叛簇扩大不能由确认失败后的幸存者筛选解释。因此，旧候选早期稳定活动背景这一解释已有明确反证。这仍不能排除更晚建立条件准稳态或其他参数存在亚稳。新计划比较b=1.10/1.11/1.12的漂移、收窄N02过渡区，合理且保持探索性质。

已验证：PYTHONPATH=code:experiments python -m unittest discover -s tests -p test_nucleation_diagnostics.py -v，7项通过。只读审查涉及计划、diagnostics、分析入口、runner、事件及生存依赖；未执行生产模拟、未修改文件，尚未审查主agent后续新增的接入测试及实际新结果。

## 处理与路由

- 数据语义检查已加入，新增提前结束拒绝测试，观察到其修改前失败、修改后通过。
- 分析指纹已覆盖全部code/*.py和experiments/*.py依赖；配置/输入任务以清单身份单列。
- 冻结提交89c9555，全套23项测试通过后启动新批次。
- 路由：supplement；保持探索性，不转入确认性论文写作。新批次完成后再次独立裁定。
