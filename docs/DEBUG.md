# Pipeline 调试指南

本页只包含可重复执行的调试入口。安装和标定分别见[环境文档](ENVIRONMENT.md)和[标定文档](CALIBRATION.md)。

## 1. 准备

```bash
cd /path/to/dice_demo
source scripts/env.sh

DBG="$DICE_ROOT/cup_grasp_demo/flow/run_debug.sh"
CFG="$DICE_ROOT/configs/green_cup.json"
RUN="$DICE_ROOT/cup_grasp_demo/datasets/green_current"
mkdir -p "$RUN"
```

运行真机命令前确认 CAN、机械臂、灵巧手和相机没有被其他程序占用。

## 2. 分阶段运行

推荐使用常驻控制台：

```bash
python3 scripts/control_console.py
```

常用命令：

| 输入 | 行为 |
| --- | --- |
| `2` | 只执行下一个阶段 |
| `3` | 连续执行到 GRIP |
| `4` | 连续执行到 SHAKE |
| `5` 或 `g` | 连续执行到 RETURN_HOME |
| `6` | 重新识别杯子并规划 |
| `1` | 查询状态 |
| `8` | 释放设备并退出 |

`control` 启动后保持 SDK、CAN、相机和模型连接。等待命令时不会重新初始化。上层 JSON 接口见[接入文档](INTEGRATION.md#常驻阶段控制供上层集成)。

一次性运行到指定阶段：

```bash
bash run.sh fast --until ready --execute
bash run.sh fast --until grip --execute
bash run.sh fast --until shake --execute
bash run.sh fast --until place --execute
```

每条 `fast` 命令均从 HOME 开始，不会续接上一个进程。

## 3. 单独测试视觉

只采集和定位绿杯，不移动机械臂：

```bash
bash "$DBG" green-detect \
  --config "$CFG" \
  --session "$RUN" \
  --show
```

主要输出：

| 文件 | 内容 |
| --- | --- |
| `green_detection.png` | 成功检测的杯口、圆心和尺寸 |
| `green_rim_debug.png` | 杯沿候选和质量诊断 |
| `runs/*/actual.json` | 结构化计算结果 |
| `runs/*/actual.log` | 本次执行日志 |

相机档位：

```bash
python3 scripts/set_camera_profile.py usb2
python3 scripts/set_camera_profile.py usb3
```

切换分辨率或裁剪后按标定文档重新标定。切换帧率后重新采图，不复用旧杯位。

## 4. 单独登记桌面

相机、桌面或基座变化后执行：

```bash
"$DICE_VISION_PYTHON" scripts/table_capture.py \
  --config "$CFG" \
  --session "$RUN"

"$CALIB_PYTHON" scripts/register_home_table.py \
  --config "$CFG" \
  --table-scene "$RUN/planar_table_scene.json"
```

完整顺序见[标定文档](CALIBRATION.md#7-应用结果并登记桌面)。

## 5. 常见问题

| 现象 | 检查项 |
| --- | --- |
| 无关节反馈 | `can0` 是否 UP；控制器是否为 CAN 模式；七轴是否使能；是否存在第二个控制进程 |
| 灵巧手无动作 | WEB 页面 Revo2 型号、灵巧手使能和 CAN 推送 |
| 相机无设备 | `lsusb`、`lsusb -t`；关闭 ffplay 或其他 RealSense 程序 |
| 标定板重复 ID | 遮住另一块板，或在板配置中设置明确 ROI/排除区域 |
| 杯沿检测失败 | 杯口无遮挡；模型掩码正确；相机内参与裁剪匹配；查看 `green_rim_debug.png` |
| 桌面记录失效 | 手眼标定哈希是否变化；重新采集并登记桌面 |
| TCP 投影位置不对 | 抓取策略文件中的法兰偏移、当前手指姿态和手眼标定 |
| 规划失败 | 杯位、关节范围、腕部参考和 TCP；不要执行旧计划 |
| 配置在运行中变化 | 等文件同步完成后重启进程，不绕过文件一致性检查 |

阶段耗时写入 `$RUN/green_pipeline_state.json`。耗时包含采集、计算、通信和电机运动，不能只用终端阶段时间判断某个函数的性能。

## 6. 反馈动作

列出、预览和执行动作：

```bash
bash run_feedback.sh --list
bash run_feedback.sh yeah
bash run_feedback.sh yeah --execute
bash run_feedback.sh thumbs-up --execute
bash run_feedback.sh tie --execute
```

`win`、`lose`、`draw` 分别是上述动作的比赛结果别名。动作完成后保持姿态，不自动回 HOME。

动作定义在 `configs/actions/gestures/`。每个动作可配置：

```json
{
  "joints_deg": [0, -80, -90, 100, 80, -5, 5],
  "hand_0_100": [0, 0, 100, 100, 100, 100],
  "speed_percent": 50,
  "finger_speed_mode": "max",
  "finger_duration_s": 0.5,
  "execution": {
    "mode": "together",
    "delay_s": 0
  }
}
```

| 字段 | 含义 |
| --- | --- |
| `joints_deg` | J1–J7 绝对角度，单位度 |
| `hand_0_100` | 六路手指目标 |
| `speed_percent` | 机械臂速度百分比 |
| `finger_speed_mode` | `max` 最大速度指令；`timed` 使用指定时长 |
| `finger_duration_s` | `timed` 模式下的手指动作时间 |
| `execution.mode` | `together`、`arm_then_hand` 或 `hand_then_arm` |
| `execution.delay_s` | 两类指令之间的软件调度延迟 |

修改动作文件后，常驻控制台输入 `r` 即可重载。动作名和别名必须在所有动作组中唯一。

六路手指顺序为：拇指尖、拇指根、食指、中指、无名指、小指。

## 7. 摇晃配置和记录

主流程摇晃配置：

```text
configs/actions/joint_shake.json
```

常用字段：

| 字段 | 含义 |
| --- | --- |
| `joints` | 参与摇晃的关节编号 |
| `amplitude_deg` | 各关节单侧幅度 |
| `velocity_deg_s` | 各关节速度预算 |
| `acceleration_deg_s2` | 各关节加速度预算 |
| `cycles` | 完整往返次数 |
| `phase_delay_deg` | 各关节相位；`null` 表示不单独配置 |
| `command_rate_hz` | `move_js()` 七轴目标更新频率 |

`command_rate_hz=200` 表示每 5 ms 尝试更新一次位置目标，不代表 200 Hz 的机械往返。实际发送情况保存在摇晃 `actual.json` 的 `command_stream`：

- `requested_hz`
- `achieved_hz`
- `max_interval_ms`
- `skipped_slots`
- `max_lateness_ms`

修改配置后重新规划并运行，不复用旧摇晃计划。

## 8. 离线 YOLO 性能测试

使用已保存的 RGB 图，不打开相机或机械臂：

```bash
"$DICE_VISION_PYTHON" scripts/benchmark_green_yolo.py \
  --config "$CFG" \
  --image /path/to/frame_000.png \
  --provider spacemit \
  --threads 2 \
  --ai-cpus 12,13 \
  --repeats 5 \
  --output "$RUN/yolo_ai_benchmark.json"
```

基准结果只表示模型加载和推理耗时，不包含相机、预处理、掩码解码和三维杯沿拟合。
