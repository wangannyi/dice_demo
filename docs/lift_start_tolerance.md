# 抬杯起点容差

`configs/green_cup.json` 的 `green_cup.lift_start_tolerance_deg` 设置抬杯计划起点与执行前关节反馈的最大偏差，单位为度。现场设置为 `0.5`，允许范围为 `0.01..0.5`；未配置时沿用 `start_tolerance_deg`。

该参数只作用于带 `allow_lift_start_drift: true` 标记的 green 抬杯请求，其他动作仍使用原起点容差。范围内直接执行已规划路径，不在 SDK 执行线程重新运行完整几何规划，也不添加固定等待。超过范围仍通过既有起点变化恢复流程重新规划。关节限位、控制状态、使能、阶段起点检查和原规划路径检查保留。

回执记录 `start_tolerance_deg` 和 `start_max_error_deg`，用于核对实际放行偏差。此容差不保证机械臂必定赶上应用倒计时；应用继续并行执行抓杯准备与倒计时。
