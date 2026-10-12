# 第二轮：背景诊断与局部几何确认

用途：记录第二轮实际证据及研究路线调整。Owner：研究执行者。
输入：[探索协议](followup_plan.md)、[N03冻结协议](confirmation_plan.md)、逐任务完成清单。
输出：本页结论、[完整探索表](followup_tables.md)和审查裁定；范围仅限指定参数与观察窗。
首轮来源见 [启动结果](results.md)，不将其样本并入本轮独立比较。

## 已完成的探索批次

| 批次 | 配置 | 数量 | 事件 | 冻结commit | campaign hash |
|---|---|---|---|---|---|
| N01 background | b=1.10/1.11/1.12，D=.30，μ=.001；L48，T20000 | 3×64=192 | 0/0/62 | `89c9555` | `6365869d6d602e52` |
| N02 refinement | b1.16，D=.35/.40，m4/8/16/32，四形状；μ0，L48，T10000 | 32×16=512 | 27 | `89c9555` | `94641bbcbf07166d` |

全部704条COMPLETE文件校验通过；种子均唯一，且与首轮1344个seed无交集。
源指纹为 `62ec03da00e6d14f`，运行时工作区干净。
分析产物：`data/nucleation/analysis/5742784fc8e58910/diagnostics.json`，包含输入清单SHA-256与分析源码指纹。

## N01：原候选的早期背景持续衰退

`b=1.12` 的64条轨迹在前2000步全部尚未确认耗竭。
合作率中位数从t100的0.938降到t1000的0.579、t2000的0.410，最大背叛簇比例中位数同期从0.008增至0.582。
同一64条轨迹的窗口平均合作率，从[100,500]到[1500,1900]下降0.387，轨迹级bootstrap95%区间为[−0.407,−0.367]。
这一早期漂移不能由已确认耗竭轨迹被排除造成。

该点20000步内62/64确认耗竭，中位确认时间5927，RMST(20000)=7394.375。
这描述的是从指定初态到确认耗竭的寿命，不能解释成数千步的成核等待。
到t10000只剩15条在险，晚期配对窗口也只保留15条；其较小漂移不能证明活动背景平稳。

相邻点提供了两个不同参照：

- b1.10：64条均未确认耗竭，后期合作率中位数约0.94；后期窗口差−0.0021，区间[−0.0058,0.0016]。数据与较平坦的高合作背景相容，但不构成准稳态存在或无限时间稳定证明。
- b1.11：64条均未确认耗竭，合作率由早期约0.965下降至后期约0.75–0.77，并出现较大背叛簇；不满足大多数轨迹的本轮高合作活动筛选。

任一个0/64格点在20000步观察窗内的单侧95%二项风险上界约4.57%，不是风险为零。
预设的活动筛选同时限制rho、资源、最大D簇与确认计数；它是描述性筛选器，不是已校准的committor恢复核心A。

## N02：找到一个可独立检验的几何协议差异

D=.35下的256条没有确认耗竭；D=.40下的256条共27次，但这些汇总混合了不同m和形状，不能视作单一参数格的风险估计。
选定m16、D=.40时，random为8/16，compact为0/16。
random是均匀无放回选择16个节点、允许偶然相邻；compact是按距离排序的紧凑集合加随机周期平移。
两协议的边界、连通分量与间距同时改变，因此任何差异都不能单独归因于一个几何指标。

## N03：独立确认

协议与分析先在commit `1c1e597` 冻结并推送，再启动512条新轨迹；campaign hash为 `ece1d6b170fb3c1c`。
两组各256次，唯一主比较random减compact，固定m16、b1.16、D=.40、μ0、L48、T10000和250步确认规则。
全部任务完成后才分析，无中途显著性检验；探索样本不进入主检验。

| 协议 | 确认耗竭 | 风险与Wilson95% | 恢复 | 未耗竭未恢复 | 次要RMST(10000) |
|---|---|---|---|---|---|
| compact | 3/256 | 1.17% [0.40%,3.39%] | 134 | 119 | 9975.70 |
| random | 149/256 | 58.20% [52.08%,64.08%] | 8 | 99 | 7966.73 |

random减compact风险差为57.03个百分点，Newcombe95%区间[50.52,62.96]个百分点；预登记双侧Fisher p=1.53736×10⁻⁵³。
Fisher按固定边际、表概率排序计算，定义见[R stats官方文档](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/fisher.test.html)；风险差区间按[statsmodels官方newcomb实现](https://www.statsmodels.org/stable/_modules/statsmodels/stats/proportion.html#confint_proportions_2indep)核对。Newcombe区间并非Fisher反演，主检验始终使用冻结的Fisher规则。

512条均通过完成清单、时钟语义及初态数组检查，每条策略损伤恰为16节点、资源损失恰为16；来源为1c1e597 / de69fecfbc13f531，工作区干净。
结果文件：`data/nucleation/analysis/71b13393bb500003/confirmation.json`；原始campaign汇总在 `data/nucleation/N03/campaigns/ece1d6b170fb3c1c/summary.json`。

独立reviewer从events.npz重数事件、按seed重建全部512张初态掩膜并用另一超几何公式重算p，均一致；裁定支持这个窄范围H1实例，无阻塞风险。

**可写的主张：** 在预先指定的参数、损伤预算与10000步观察窗内，均匀无放回选择损伤节点的协议，比所测试的紧凑斑块协议产生更高的持续宏观耗竭概率。

**不能写的主张：** 分散损伤总是更危险、周长是唯一因果变量、已经证明自发成核/单核逃逸、已经获得临界核或无限时域寿命。

## 验证与范围

当前28项测试通过；本轮新增1216条正式轨迹（192+512+512），连同首轮共2560条。诊断新增了landmark事件边界、当前持续段、局部簇活动门槛、配对窗口、缺失采样、提前技术终止拒绝；N03另测整数权重Fisher、零计数风险差区间、协议偏离拒绝和短程等成本运行。
原物理内核、晶格、旧观测函数和原data/main接口没有改变。
此外使用已有环境作一次性参考核对：1936张小表的Fisher与SciPy1.11.4一致，5组边界/不对称风险差区间与statsmodels0.14.6一致；没有新增运行依赖。
已生成并查看确认图 `figures/nucleation/confirmation/confirmation.png` / `.pdf`，以及探索二维图 `figures/nucleation/followup/followup.png` / `.pdf`。环境仍有与本图无关的Axes3D导入警告，未修改全局依赖。

## 复算

```bash
PYTHONPATH=code:experiments python -m unittest discover -s tests -v
PYTHONPATH=code:experiments python experiments/analyze_nucleation_followup.py --background configs/nucleation/n01_background.json --seeds configs/nucleation/n02_refinement.json
PYTHONPATH=code:experiments python experiments/plot_nucleation_followup.py data/nucleation/analysis/5742784fc8e58910/diagnostics.json
PYTHONPATH=code:experiments python experiments/analyze_nucleation_confirmation.py --config configs/nucleation/n03_geometry_confirmation.json
PYTHONPATH=code:experiments python experiments/plot_nucleation_confirmation.py data/nucleation/analysis/71b13393bb500003/confirmation.json
```

分析代码版本变化会产生新analysis hash；上述固定路径指本轮记录版本。重跑模拟应使用对应冻结commit，避免不同来源复用同一任务。

## 下一阶段与停止条件

1. H1当前窄主张已通过独立确认，不继续为同一比较追加样本。下一步若扩展，须新增独立参数/损伤预算或空间路径对照，检验适用范围与机制。
2. H3仍无完整证据。b1.10可作为活动背景候选；需在其邻域寻找兼具较稳定背景与可测逃逸的区间，再做准备期及不同初态检验。b1.12当前初态协议的长确认寿命不再直接作为稀有成核等待证据。
3. 未校准A核心、完整成功核谱系或committor；暂不启动N04全尺度扫描、N06/N07或宣称Avrami机制。

证据裁定见 [claim_verdict.md](claim_verdict.md)，独立审查记录在 [review-traces/followup01/](review-traces/followup01/)。
所有原始数据、来源压缩包和旧分析版本均保留；日志与开发产物在tmp/，未删除用户资料。原始数据约1.7GB留在本地，不推入Git。
