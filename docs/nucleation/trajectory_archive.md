# 原始轨迹归档与恢复

用途：从GitHub取得完整逐运行数据，校验后恢复到独立目录。
Owner：研究执行者。输入：本地N01/N02/N03全部文件；输出：三个不可变归档、文件清单和恢复步骤。
范围：截至2026-10-06的已有数据；没有重新模拟、筛选或改写科学结果。
相关：[研究结果](growth_results.md)、[证据裁定](claim_verdict.md)、[文档入口](../README.md)。

## 内容与计数

归档位于[`data/nucleation/archives/2026-10-06/`](../../data/nucleation/archives/2026-10-06/)，通过[Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)存储。

| 归档 | 完整逐运行目录 | 内容 |
|---|---:|---|
| `N01.tar.gz` | 962 | N01全部轨迹及campaign记录 |
| `N02.tar.gz` | 2176 | N02全部轨迹及campaign记录 |
| `N03.tar.gz` | 512 | N03全部轨迹及campaign记录 |
| 合计 | 3650 | 每条包含快照、轨迹、簇/谱系、事件、配置、来源和COMPLETE清单 |

三个包合计1,931,826,311字节（约1.80GiB），完整保存77,362个文件。

**3650是归档目录数，正式研究统计仍为3264条。** 额外386条均保留：N01旧格式配置`cd869fba9bb17b92`的384条，以及测速配置`3955d2a20a01b214`的2条；它们没有因此被并入正式主张的统计样本。
正式N01首轮引用的是`b9b63239ce022531`；旧格式配置缺少后续添加的`schema_version`与`study`字段，与正式配置身份不同。
分析汇总、图、冻结协议和运行源码压缩包已分别在Git中保留；源码归档见[`_sources/`](../../data/nucleation/_sources/)。

- `manifest.json`：各包字节数、SHA-256、文件数、运行目录数和归档时的仓库提交；各历史运行的源码身份仍以其provenance记录为准。
- `SHA256SUMS`：三个压缩包的校验和。
- `FILES.sha256`：所有解压后文件的校验和，路径相对于仓库根目录。
- [`REMOTE_VERIFICATION.json`](../../data/nucleation/archives/2026-10-06/REMOTE_VERIFICATION.json)：独立空缓存下载后的远端实体校验记录，注明已验证的归档提交和各包SHA-256。

归档前逐条验证COMPLETE；归档后逐成员解压并核对源文件SHA-256。

## 本地状态（2026-10-08更新）

本仓库工作树的本地逐运行目录已于2026-10-08清理：
- 截至2026-10-06的原始轨迹完整保存在上述LFS归档中，需要时按[下载与校验](#下载与校验)恢复，数据无损。
- 2026-10-07/08论文相图campaign（`243807bdd1e9a2c1`、`dd9cf1618cf34594`）的原始轨迹在归档前已被删除且未另行备份，无法恢复。其campaign规格（含种子）、分析输出与论文图表均已入库；如需原始轨迹，须按`campaigns/`中的配置与种子重跑再生。
归档文件名固定日期；将来新增数据使用新的日期/批次，不覆盖既有归档。

## 下载与校验

需要已安装Git LFS。以下命令面向一个新克隆的研究仓库：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --single-branch --branch main git@github.com:BKZhao/Stochastic-Nucleation-SEG.git stochastic-nucleation-restore
cd stochastic-nucleation-restore
git lfs install --local
git lfs pull origin --include='data/nucleation/archives/2026-10-06/*.tar.gz' --exclude=''
(cd data/nucleation/archives/2026-10-06 && sha256sum -c SHA256SUMS)
```

新克隆中的`origin`指向本研究仓库。现有开发工作树有两个remote，应使用`research`下载，不对原物理模型的`origin`操作。
不安装/拉取LFS时，Git中的归档可能只是很小的指针文件，不能直接解压。GitHub的普通“Download ZIP”是否含LFS实体取决于仓库设置，优先使用上述命令。

## 恢复到独立目录

压缩包校验通过后，在仓库根执行；目标已存在时整段停止，请换用新的恢复目录，避免覆盖已有研究数据。

```bash
(
    set -eu
    mkdir -p tmp
    mkdir tmp/trajectory-restore-20261006
    for stage in N01 N02 N03; do
        tar -xzf "data/nucleation/archives/2026-10-06/${stage}.tar.gz" -C tmp/trajectory-restore-20261006
    done
    cd tmp/trajectory-restore-20261006
    sha256sum --quiet -c ../../data/nucleation/archives/2026-10-06/FILES.sha256
)
```

解压得到`tmp/trajectory-restore-20261006/data/nucleation/N01`等目录。
分析时可将`--root`指向该独立的`data/nucleation`。完整场绘图中保存的绝对运行路径属于原机器，跨机器使用时需据对应原始数据位置重建分析报告。
来源记录中的历史HEAD、dirty状态和路径按运行时原样保存；后续上传不会把旧运行伪装为当前源码的产物。
