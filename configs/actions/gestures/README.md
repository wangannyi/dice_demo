# 猜拳出拳参数

`rps.json` 的四个动作保持原有固定关节端点，默认速度 100%。

- `arm_delivery: smooth_profile`：当前板端使用的默认方式。通过 MoveJS 上位机平滑插值到达固定端点；保留实时反馈、关节范围与到位检查。
- `controller_endpoint` 暂不用于现场猜拳：2026-09-28 实测单次 MoveJ 下发后 J3 未从 HOME 移动到预备端点，触发到位超时。仅零位移调用成功不能证明该模式可用，恢复前必须验证非零位移运动。
- `execution.mode: together`：臂移动时并行调度手指。
- 三种拳形 `execution.delay_s: 0.9`：从臂发送运动命令起算，0.9 秒后开始手指成形。预备动作 `rps-ready` 保持 0 秒。
- `finger_speed_mode: max`：指令使用手指最大速度档位，`finger_max_wait_s: 0.65` 仅为观察等待时间，不是实际到位测量。

上层 main 仓库先调用 prepare_throw（HOME 检查 → rps-ready），成功后同时播放口令并调用已准备好的 throw_gesture。默认旧调用仍执行完整预备＋出拳链。动作失败时上层显示机械臂的实际错误，不进入人手识别来代替动作诊断。

现场口令与运动同步还受浏览器音频播放延迟影响，可以通过 hand delay 调整手势形成时间；不能通过跳过预备动作或取消到位检查实现同步。
