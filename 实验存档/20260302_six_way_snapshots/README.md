# 六组实验快照存档

本目录统一存放以下六组实验的结果快照与代码快照：

1. `pre_performance`
2. `pre_physics`
3. `post_performance`
4. `post_physics`
5. `mi_performance_100`
6. `mi_physics_100`

每组包含：
- `result_snapshot/`：数据、结果图、日志、特征配置等快照
- `code_snapshot/`：对应实验阶段的关键脚本快照
- `manifest.json`：该组快照来源说明

说明：
- `pre_*` / `post_*` 的代码快照主要由 Git 历史版本恢复。
- `mi_*` 的代码快照来自当前 MI 分支工作树版本。
- `mi_performance_100` 的 `results/` 目录在首次归档时发生复制异常，因此该组保留了 `final_results/`、`logs/`、`data/`、`feature_config.py` 以及六组汇总表作为完整补充依据。
