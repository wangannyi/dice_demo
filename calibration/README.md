# 标定工具

在仓库根目录使用统一入口，参数编辑 `configs/calibration_workflow.json`：

```bash
bash calibrate.sh first
bash calibrate.sh auto --execute
bash calibrate.sh restore --execute
```

分别用于首次人工示教标定、HOME 后自动重采标定、HOME 后通过固定板恢复外参。首次示教仍由现场人工操作机械臂。

仓库内置的 20 姿态示教轨迹位于 [`trajectories/handeye_auto_teach_covered_20260922`](trajectories/handeye_auto_teach_covered_20260922/)。配置已指向该轨迹；首次运行 `bash calibrate.sh plan` 生成与当前机器路径及文件哈希绑定的新计划。

`bash calibrate.sh apply` 备份并应用最近结果，重新采集和登记桌面；`status` 查看最近数据；`--dry-run` 仅预览。详细步骤、固定板登记、质量门槛和回滚见 [标定指南](../docs/CALIBRATION.md)。

`calibrate.py`、`auto_collect.py`、`reference_board.py` 等底层入口保持兼容，参见 [底层命令](../docs/CALIBRATION_ADVANCED.md)。
