# 猜拳出拳参数

`rps.json` 的四个动作保持原有固定关节端点，默认速度 100%。

- `arm_delivery: controller_endpoint`：通过 MoveJ 单次发送目标，使用控制器关节轨迹规划；实时检查反馈、关节范围、路径包络及到位超时。
- `arm_delivery: smooth_profile`：原有 MoveJS 上位机插值，其他手势默认仍使用此路径。
- `execution.mode: together`：臂移动时并行调度手指。
- 三种拳形 `execution.delay_s: 0.9`：从臂发送运动命令起算，0.9 秒后开始手指成形。预备动作 `rps-ready` 保持 0 秒。
- `finger_speed_mode: max`：指令使用手指最大速度档位，`finger_max_wait_s: 0.65` 仅为观察等待时间，不是实际到位测量。

上层 main 仓库先调用 prepare_throw（HOME 检查 → rps-ready），成功后同时播放口令并调用已准备好的 throw_gesture。默认旧调用仍执行完整预备＋出拳链。

固定目标模式保留原关节端点和路径检查，不向 MoveJS 发送大幅跳点。现场完整口令与运动同步需结合音频播放延迟验证；可以通过 hand delay 调整手势形成时间。
