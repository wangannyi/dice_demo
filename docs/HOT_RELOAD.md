# 参数热加载

上层游戏使用的 `bash run.sh control --execute` 常驻进程默认开启热加载。
修改并保存下列 JSON 后，下次独立动作或新一轮抓杯开始前自动校验并生效，无需重启应用、重新连接 CAN 或手动发送 reload。

| 文件 | 常用参数 | 生效边界 |
| --- | --- | --- |
| `configs/green_cup.json`（或 `DICE_CONFIG` 指定的主配置） | `green_cup.fast_speed_percent`、`fast_phase_speed_percent`、`place_offset_base_mm`、`fast_finger_duration_s`、`joint_delivery` 等 | 下次独立动作／新一轮 HOME 前 |
| `vision/strategy/green_cup.json`（由 `green_cup.strategy_file` 指定） | 抓取偏移 `contact_offset_base_mm`、抬杯高度 `lift_mm`、抓握／松手目标等 | 新一轮 HOME 前 |
| `configs/actions/joint_shake.json`（由 `green_cup.joint_test_config` 指定） | 摇晃幅度、速度、加速度、周期数等 | 新一轮 HOME 前 |
| `configs/actions/gestures/*.json` | 猜拳、反馈和 HOME 手势的关节目标、手指目标、速度、延时、执行模式；支持新增／删除组文件 | 下次独立动作前 |

这些路径是默认位置；实际读取主配置指定的策略和摇晃文件。
策略文件的同名字段仍遵循现有优先级：主配置 `green_cup` 显式值优先。
FAST 流程仍使用 `green_cup.fast_speed_percent`，不是顶层 `speed_percent`；独立手势仍使用手势自身的 `speed_percent`。热加载不改变参数含义和控制器限速。

## 使用方式

例如修改 `configs/green_cup.json` 内 `green_cup.place_offset_base_mm`，保存后正常再玩一局，下一轮会使用新的放杯偏移。不需要在网页刷新或重启服务。

`HOME → … → RETURN_HOME` 是一个完整参数周期。即使上层分多次 `advance` 调度，途中保存的新参数也等当前周期结束后再生效。正在运行的动作使用固定快照，包括后台预规划和摇晃轨迹；不会在持杯期间切换参数。

需要主动校验时，在控制台输入 `f`，或向常驻进程发送：

```json
{"id":"reload-1","command":"reload"}
```

只允许流程空闲时执行。默认自动热加载模式下，它同时重载主配置、策略、摇晃和手势。成功输出 `config_reloaded`（变化文件、动作名）及 `actions_reloaded`；自动加载只输出 `config_reloaded`。上层继续等待原命令的 `action_completed`／`command_completed`，不要将重载事件当成动作完成。

JSON 未保存完整、参数越界、手势冲突等会返回 `rejected`，`code=reload_failed`。旧参数保留，本次命令不执行，也不会因此触发自动归位。修正并保存文件后重试即可。建议编辑器使用原子保存；一次修改多个文件时，保存完全部文件再开始下一局。

## 资源与安装配置

普通运动参数变化复用 SDK 和相机连接，并清除旧预规划、姿态和采集缓存。感知参数或 RTSP 设置变化时，会关闭旧视觉资源，并在下一次 HOME 重新初始化；因此这类修改后的第一轮可能多等一会儿。

以下变更仍需停止应用、完成相应核对后重新启动：

- CAN 通道、pipeline 类型、`persistent_runtime`、热加载开关。
- `vision/camera.json` 的相机参数。
- 标定、HOME 姿态文件 `configs/actions/home.json`、桌面登记、参考姿态、TCP 候选、模型文件及其路径。
- Python 程序代码。

标定与桌面绑定校验、碰撞检查、真实控制器限速等保持有效。手势库中的 `home` 可以热加载，但应与 HOME 姿态文件一致；两处不同仍会提示警告。

如需保留旧的固定配置模式，在主配置的 `green_cup` 中设置 `"hot_reload": false` 并重启。关闭后 `reload` 仍只重载手势表，主配置修改继续触发运行期间变更保护。

`fast` 单次命令、标定 CLI 等在每次启动时读取配置；本功能针对上层游戏使用的常驻 `control` 进程，不会使所有独立工具都变成文件监听服务。
