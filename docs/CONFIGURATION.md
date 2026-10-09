# 参数配置参考

本页列出绿杯 Pipeline 的常用配置、文件分工及运动模式。安装和运行入口见 [README](../README.md)，参数生效时机见[热加载说明](HOT_RELOAD.md)。

## 主配置字段

主配置为 [`configs/green_cup.json`](../configs/green_cup.json)。

| 字段路径 | 含义 |
| --- | --- |
| `serial`、`channel` | RealSense 序列号和 CAN 接口 |
| `calibration` | 当前安装的手眼标定结果 |
| `home` | HOME 七轴姿态 |
| `green_cup.home_table_scene` | 已登记的桌面平面 |
| `green_cup.installation_requires_calibration` | `true` 时禁止真机 Pipeline，需先完成标定和桌面登记 |
| `green_cup.strategy_file` | 抓取策略文件；其中包含 TCP、接触点、腕部、抬杯和手指参数 |
| `green_cup.fast_speed_percent` | FAST 普通运动速度百分比 |
| `green_cup.fast_phase_speed_percent` | 指定阶段的速度覆盖值 |
| `green_cup.fast_finger_duration_s` | FAST 抓握／松手时长，单位秒；当前配置为 `0.3` |
| `green_cup.place_offset_base_mm` | 放杯目标相对抓取位置的基座坐标系 `[X, Y, Z]` 补偿，单位 mm |
| `green_cup.perception` | 绿杯模型、尺寸、杯沿和推理后端 |
| `green_cup.joint_test_config` | 摇晃动作配置文件 |

六路手指顺序为：拇指尖、拇指根、食指、中指、无名指、小指。TCP 偏移使用法兰坐标系，不是图像坐标系。

## 配置文件分工与优先级

| 文件 | 修改内容 |
| --- | --- |
| [`configs/green_cup.json`](../configs/green_cup.json) | 硬件接口、标定和 HOME 路径、运动速度、手指时长、放杯补偿、感知和控制器限制 |
| [`vision/strategy/green_cup.json`](../vision/strategy/green_cup.json) | 杯子抓取参数：接触点／TCP 偏移、腕部参考角、抬杯高度、六路抓握／松手目标及路径间距 |
| [`vision/camera.json`](../vision/camera.json) | 相机分辨率、帧率和裁剪 |
| [`configs/actions/joint_shake.json`](../configs/actions/joint_shake.json) | 摇晃动作的关节、幅度、速度和周期 |
| [`configs/actions/gestures/`](../configs/actions/gestures/) | 猜拳和胜负反馈动作 |

`green_cup.strategy_file` 指向策略文件。加载时，策略字段只补充主配置 `green_cup` 中缺少的字段；同名字段以主配置为准。参数只保留一处定义：例如修改抓握／松手时长时，修改主配置的 `green_cup.fast_finger_duration_s`，不要再在策略文件中添加同名字段。

当前绿杯 Pipeline 始终使用 FAST 参数：普通运动读取 `green_cup.fast_speed_percent`；手指时长优先读取 `green_cup.fast_finger_duration_s`，缺省时才使用策略中的 `finger_duration_s`。运行时固定 `finger_settle_s = 0`、`require_arm_position = false`，因此这些策略字段不能用于增加 FAST 阶段等待。

主配置中的动作调参和策略中的抓取参数支持热加载，在下次独立动作或新一轮 HOME 前生效，不会修改正在执行的轨迹。相机、标定和 HOME 等受保护配置的生效规则见[参数热加载说明](HOT_RELOAD.md)。

## 摇晃动作

[`configs/actions/joint_shake.json`](../configs/actions/joint_shake.json) 配置参与关节、幅度、速度、加速度、周期和目标更新频率。`command_rate_hz` 是七轴位置目标的发送频率，不是杯子的往返频率。实际频率受行程、轨迹和控制器限制。

## 放杯模式

以下为 `configs/green_cup.json` 中 `green_cup` 对象的相关字段节选；该对象的其他字段保持原值：

```json
{
  "lower_command_mode": "controller_endpoint",
  "fast_phase_speed_percent": {"approach": 100, "lower": 100, "return_home": 100}
}
```

`lower_command_mode` 只作用于 LOWER 放杯阶段。`controller_endpoint` 将规划、检查后的关节目标通过 `move_j` 下发，由控制器规划加减速；`smooth_profile`（省略时的默认值）恢复主机插值 `move_js`。`lower` 为放杯速度百分比，范围 1–100；省略时沿用普通 FAST 速度。当前配置中这三个阶段均为 100，普通 FAST 速度为 30。

保持放杯目标、桌面间隙检查、关节限制和到位检查；不会向桌面下方增加目标偏移，不保证产生撞击或一定消除骰子倾斜。其他阶段和放杯纠偏保持原运动模式。恢复原放杯行为可将模式改为 `smooth_profile`、`lower` 改为 30。配置在下一轮热加载；程序文件修改后需重启应用。

放杯固定端点模式每 50 ms 重发同一组七轴目标，直到全部关节到位或超时，避免只发送一组 CAN 帧后部分关节未更新。保留到位门槛和超时退出，执行记录包含发送次数、最终反馈及剩余误差。其他阶段不启用此重发。

## 文件检查与起点偏差

`green_cup.fast_file_check_interval_s` 设置程序/安装文件检查的最小间隔（0–1 秒，默认 0，即每次检查）；现场设为 0.25，合并阶段入口与动作下发的重复文件扫描。每轮开始、独立动作前和显式热加载时强制检查；普通阶段的文件变化在间隔结束后的下一次检查发现。此参数不影响机械臂实时反馈检查。

`joint_delivery.start_drift_tolerance_deg` 为读取限位期间允许的起点变化（0.1–0.5°，默认 0.1°）；现场设为 0.25°。允许范围内按最新关节反馈重算运动起点及速度曲线，超过范围仍拒绝动作。通信、关节限制、运动范围及超时检查保持启用。`precision_error_action: record` 和 `lower_recovery_attempts: 0` 保持现有设置。
