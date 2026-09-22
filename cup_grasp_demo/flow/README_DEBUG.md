# 银杯对点与独立调试

当前绿杯分步运行、关节测试、平面摇晃及故障排查见 [统一调试文档](../../docs/DEBUG.md)。

## 历史银杯对点

从仓库根目录执行，先设置环境。该配置只适用于银杯侧抓实验，不能替代当前绿杯配置。

```bash
source scripts/env.sh
DBG="$DICE_ROOT/cup_grasp_demo/flow/run_debug.sh"
CFG="$DICE_ROOT/cup_grasp_demo/flow/index_joint_center/config.json"
RUN="$DICE_ROOT/cup_grasp_demo/datasets/compare_current"
"$DBG" capture --config "$CFG" --session "$RUN" --show
"$DBG" plan --session "$RUN" --frame tcp --gap-mm 0 --cup-removed --output "$RUN/tcp_plan.json"
```

`--frame flange` 把法兰原点作为比较点；`--frame tcp` 按法兰到 TCP 的变换补偿。`--cup-removed` 表示操作者已移开杯子、保留原目标空间，程序不会自动移杯。

确认实物已移开并检查计划后执行：

```bash
"$DBG" move --plan "$RUN/tcp_plan.json" --show --execute
```

到位收据记录模型与关节反馈误差；实物接触点仍需独立测量。不要用机械臂自身 FK 结果证明手眼标定精度。
