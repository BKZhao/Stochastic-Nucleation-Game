# data/ 目录口径（2026-10-07 整理）

## 结构与体量（整理后）

| 目录 | 体量 | 内容 | git |
|---|---|---|---|
| `N06/` | ~26G | 成核研究全部逐运行目录（方向/机制/稳健性/R3 各轮 campaign 的 seed-* 目录，每目录含 trajectory/clusters/events npz + states/ 快照） | 仅 `campaigns/`（96 文件） |
| `N02/` | 7.4G | 母态银行逐运行目录（含密快照银行 `0b0d50e0b1a4711a` 的 250 间隔快照） | 仅 `campaigns/` |
| `N01/`, `N03/` | <0.5G | 早期诊断与确认轮 | campaigns + 归档 |
| `analysis/` | 5.8M | 全部分析 JSON（direction/mechanism/robustness/r3/coupling0/post_confirmation，按 identity 哈希命名） | 全部追踪 |
| `_sources/` | 3.9M | 每次运行的源码指纹压缩包 | 全部追踪 |
| `archives/` | 11M | 原始轨迹 Git LFS 归档（3,650 运行目录，校验和见 trajectory_archive.md） | LFS |
| `main/`（若存在） | ~7G | **原论文数据，不属于本研究**，按原样保留 | 不追踪 |

## 隔离归档（2026-10-07）

机制轮首跑期间因源码中途修正产生的 241 个 `.quarantine-*` 目录（1.8G，含 v1/v2 源码版本的部分完成运行）已打包为 `tmp/quarantine_mechanism_attempts.tar.gz`（1.3G，sha256 `21b20402…b35ca33`）后从 data/ 删除。正式数据不含任何隔离运行——机制轮最终 3,840 条全部为单一源码版本（`_sources/bec666d636a8da4d.tar.gz` 及 `f0c002c24cfde2a6.tar.gz` 链）。

## 删除/压缩决策

- **逐运行目录不压缩、不删除**：论文复现包要求"正文某个点 → 对应母态与处理臂 → 分支结果 → 运行分析"可直接走通；磁盘充足（3.3T 空闲）。
- **quarantine 打 tar 后删除**：审计价值保留（tar + 校验和在 tmp/），日常路径上不再出现隐藏目录。
- 大批量归档（>10G 压缩）留待论文投稿后的 LFS 批次，按 [trajectory_archive.md](trajectory_archive.md) 协议执行。
