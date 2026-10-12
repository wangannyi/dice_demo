# 猜拳出拳参数

`rps.json` 的四个动作保持原有固定关节端点，默认速度 100%。

- `rock`、`paper`、`scissors` 使用 `arm_delivery: bounded_blend`：七轴共用梯形基准轨迹的进度，通过 MoveJS 连续发送带受限提前量的目标。首尾提前量为零，关节终点不变，保持反馈、关节范围与单步不超过 1° 的检查。
- 每个出拳动作的 `arm_blend.endpoint_blend` 为混合强度（0～0.2，当前 0.2）；`arm_blend.endpoint_lead_max_deg` 为相对重新计时后的基准曲线的最大提前角度（0～1°，当前三个动作均为 0.2°）。提前量作用于统一进度，不能让某一关节单独抢跑。它不是 SHAKE 的无限幅 `endpoint_blend`，不能照搬 SHAKE 的 0.8。
- 混合模式发送周期不大于 5 ms（理想调度至少 200 Hz；实际频率受系统调度影响）。曲线会按混合后的速度、加速度上界重新计时，保留原预算，因此不承诺缩短总动作时间。日志记录 `base_duration_s`、`duration_s`、`blend_time_scale` 和实际发送耗时，供现场对照。
- `rps-ready` 仍使用 `smooth_profile`，预备动作不增加提前量。要取消出拳混合，只需将对应动作的 `arm_delivery` 改回 `smooth_profile`；`arm_blend` 此时不参与执行。
- `controller_endpoint` 仍可显式选择，使用 MoveJ，并在未到位时每 50 毫秒重发同一个七轴终点。重发可补偿部分目标未生效，但不能保证现场控制器七轴同步启动，因此不作为当前猜拳配置。
- `execution.mode: together`：臂移动时并行调度手指。
- `rps-ready` 抬臂同时张开六路手指，目标为 `[0, 0, 0, 0, 0, 0]`，不再执行预备握拳序列。
- 三种拳形及预备动作 `rps-ready` 均使用 `execution.delay_s: 0`：在第一个臂运动控制周期内发送手指命令，臂和手一起启动。
- `rock` 从张手状态开始，先闭合四指，间隔 `hand_sequence.interval_s` 秒后再闭合拇指，最终目标仍为 `[100, 100, 100, 100, 100, 100]`。间隔沿用各现场配置；调整时应核对四指闭合与拇指收拢的时序。
- `finger_speed_mode: max`：指令使用手指最大速度档位，`finger_max_wait_s: 0.65` 仅为观察等待时间，不是实际到位测量。

上层 main 仓库先调用 prepare_throw（HOME 检查 → rps-ready），成功后播放口令，收到浏览器真实起播回执后等待 1.55 秒，再调用已准备好的 throw_gesture。原来的 0.9 秒手指延时移入共同启动延时（0.65 + 0.9 = 1.55 秒），保留原手指启动节拍，将手臂启动推迟到同一时刻。默认旧调用仍执行完整预备＋出拳链。动作失败时上层显示机械臂的实际错误，不进入人手识别来代替动作诊断。

现场微调节拍时，修改 main 仓库 `backend/games/rps/manifest.json` 的 `play.on_enter[0].on_start[0].delay_seconds`，臂手一起提前或推迟。保持本配置的手指延时为 0，避免再次先动臂后动手。实际启动还包含动作初始化与通信耗时，需要现场核对；不跳过预备动作或到位检查。
