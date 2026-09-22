# Pipeline 分步调试

## 1. 准备入口

在仓库根目录执行。环境安装见[顶层 README](../README.md)，重新安装设备后先完成[标定](CALIBRATION.md)。

```bash
source scripts/env.sh
DBG="$DICE_ROOT/cup_grasp_demo/calibration_debug/run_debug.sh"
CFG="$DICE_ROOT/configs/green_cup.json"
RUN="$DICE_ROOT/cup_grasp_demo/datasets/green_current"
mkdir -p "$RUN"
"$DBG" config-check --config "$CFG"
```

`green_current` 保存最近一次定位与计划，`runs/` 保存每次执行记录。修改参数后重新运行定位和规划，不能把旧计划当成新配置的结果。

## 2. 逐阶段执行

```bash
./run.sh step --show --execute
```

每阶段按 Enter 继续，输入 `q` 退出。预览窗口显示时间有限，不需要一直等窗口关闭。`--show` 用于 STEP；FAST 不生成这些非必要图像。

| 阶段 | 操作与观察 |
| --- | --- |
| HOME | 张手、归位；检查实际张开姿态 |
| CAPTURE | 定位杯口，确认圆心、杯高与轮廓匹配 |
| PLAN | 根据杯口位置、TCP 和抓取偏移求解关节路径 |
| APPROACH | 到抓取位；查看 TCP 投影与实际中指指根是否一致 |
| GRIP | 发送六路闭手目标，确认实物持杯情况 |
| LIFT | 抬升配置距离 |
| SHAKE | 按 `configs/joint_shake.json` 执行往复 |
| LOWER | 放回抓取时的放置高度 |
| OPEN | 张手放杯 |
| RETURN_HOME | 返回 HOME |

只运行到抓取位置或闭手：

```bash
./run.sh step --until ready --show --execute
./run.sh step --until grip --show --execute
```

这两条是分别从 HOME 开始的新流程，不是从上次暂停处续跑。完整流程用 `--until place`，其中包含最后返回 HOME。发生错误时先读实际状态和日志，不能假设程序退出就已经放杯。

## 3. 单独检测杯子

```bash
"$DBG" green-detect --config "$CFG" --session "$RUN" --show
```

此命令只采集、识别和计算，不移动机械臂。重点查看：

- `green_detection.png`：成功检测叠加图。
- `green_rim_debug.png`：杯沿诊断图。
- `green_tcp_current.png`：STEP 阶段的 TCP 投影。
- `runs/` 下的 `actual.log`、`actual.json`：执行器日志和结果。

TCP 图中的点来自关节反馈、模型和标定变换，不能当成相机直接测出的物理接触点。调整 `green_cup.tcp_offset_flange_mm` 时，三个数沿法兰坐标轴，不是图像左右上下方向。

## 4. 单独关节摇晃

```bash
JT="$DICE_ROOT/cup_grasp_demo/calibration_debug/run_joint_test.sh"
JCFG="$DICE_ROOT/configs/joint_shake.json"
"$JT" plan --config "$JCFG" --system-config "$CFG" --session "$RUN" &&
"$JT" run --plan "$RUN/joint_plan.json" --execute
```

从当前姿态生成往复，不包含抓杯。`joints` 与 `amplitude_deg`、`velocity_deg_s`、`acceleration_deg_s2` 按下标一一对应。幅度是单侧角幅，完整行程为两倍；`cycles` 是完整往返次数。规划频率不是实测频率。

当前交付配置使用 J1/J4/J5/J6/J7、各 ±5°、6 个周期。改动前查看实际配置，不要沿用早期 J1/J4/J7 的描述。`--execute` 直接执行，不再输入确认词；规划未通过时先解决报错。

## 5. 单独平面摇晃

独立桌面采集不识别杯子、不移动机械臂。机械臂、桌面或相机变动后重采：

```bash
JS="$DICE_ROOT/cup_grasp_demo/calibration_debug/run_planar_shake.sh"
JSCFG="$DICE_ROOT/cup_grasp_demo/calibration_debug/planar_shake.json"
"$JS" table-capture --config "$CFG" --session "$RUN"
"$JS" plan --config "$CFG" --trial-config "$JSCFG" \
  --session "$RUN" --table-scene "$RUN/planar_table_scene.json"
```

确认是空手测试后执行：

```bash
"$JS" run --plan "$RUN/planar_js_plan.json" --load empty --execute
```

此入口与 GREEN PIPELINE 的关节摇晃不同。不要用 `--load empty` 描述实际持杯状态。历史法兰/TCP 对点工具见[原调试工具说明](../cup_grasp_demo/calibration_debug/README_DEBUG.md)，新绿杯流程以本文和顶层配置为准。

## 6. 耗时和故障定位

`[耗时]` 是阶段总耗时，可能包含连接、采集、计算、发送与运动。STEP 的图像采集和显示也计入阶段；比较性能时使用相同配置的 FAST。

| 现象 | 优先检查 |
| --- | --- |
| CAN 无反馈 | `ip -details link show can0`、控制器 CAN 模式和推送、是否存在另一个执行器 |
| 手无动作 | WEB 的灵巧手使能、手部型号与当前反馈；机械臂使能不等于手部使能 |
| `table_plane_not_supported` | 新标定是否应用、桌面数据是否重采、有效深度和工作区 |
| 杯沿失败 | 诊断图、杯口无遮挡、固定/测量高度模式、相机分辨率与内参一致性 |
| TCP 不在期望部位 | TCP 基础变换、法兰偏移、手指张开状态与参考姿态 |
| 规划失败 | 目标位置、关节范围、姿态参考、当前配置；不要执行旧计划代替 |

不要同时启动两个真机控制程序。错误记录需要保留实际原因，不能把发送命令成功写成实际抓牢或放置成功。

## 7. YOLO 两核 AI 离线测试

使用已保存的 RGB 图比较后端，不启动相机或机械臂：

```bash
"$DICE_VISION_PYTHON" scripts/benchmark_green_yolo.py \
  --config "$CFG" --image /实际路径/frame_000.png \
  --provider spacemit --threads 2 --ai-cpus 8,9 --repeats 5 \
  --output "$RUN/yolo_ai_benchmark.json"
"$DICE_VISION_PYTHON" scripts/benchmark_green_yolo.py \
  --config "$CFG" --image /实际路径/frame_000.png \
  --provider cpu --threads 4 --repeats 5 \
  --output "$RUN/yolo_cpu_benchmark.json"
```

JSON 区分模型加载、首次推理和后续纯推理时间；同名 NPZ 保存检测掩码。`task_affinities` 可核实 AI 线程绑定到 8、9。纯推理耗时不含相机采集、预处理、掩码解码或三维拟合，也不等于 CAPTURE 总耗时。

后端设置参见[SpacemiT 官方说明](https://github.com/spacemit-com/docs-ai/blob/main/en/compute_stack/ai_compute_stack/onnxruntime.md)。切换后端需比较检测数、掩码与三维定位结果，不能只看速度。

## 8. 相机取帧耗时

FAST 默认预热 5 组 RGB、深度、左右红外均已更新的有效帧（重复帧不计数），正式采集前额外丢帧为 0。配置项为 `green_cup.fast_camera_warmup_frames` 和 `green_cup.fast_camera_fresh_discard_frames`。相机启动预热不是保存采样帧数；固定高度 FAST 仍只保存 1 帧。STEP 和通用采集默认保留 20 帧预热。

每次采集的 `rgbd/frame_000.json` 中，`capture_timing` 记录流启动、预热、新鲜帧等待及采集处理耗时。清理旧队列后直接使用下一组 RGB/深度均已更新的帧，不再固定跨过两帧。5 帧不代表所有光照下曝光都已稳定；若新场地初始画面偏暗，可增加预热帧数。

FAST 的桌面拟合不通过或工作区杯子候选不唯一时，复用相机取一帧重试，最多一次；正常路径不增加等待。重试原因保存为 `green_capture_retry.json`。重试仍失败时退出，不使用上次成功定位。

## 9. 比大小后的反馈手势

独立入口 `run_feedback.sh` 不运行抓杯流程、不采集相机或调用 YOLO。先完成放杯，再执行反馈动作；当前 yeah、thumbs-up 配置为机械臂与手指同时启动，时延为 0；也可配置先后执行。完成后保持姿态，不自动返回 HOME。保留现有 CAN 控制、关节路径和桌面检查，使用当前标定对应的已保存桌面记录。

```bash
cd /home/test2/dice_demo
# 仅查看目标，不连接硬件
bash run_feedback.sh yeah
bash run_feedback.sh thumbs-up
# 直接执行，不再输入确认词
bash run_feedback.sh yeah --execute       # 机械臂赢了
bash run_feedback.sh thumbs-up --execute  # 机械臂输了
```

也可使用 `win` / `lose`，胜负均以机械臂为视角。此脚本未自动订阅比大小结果；上层程序在最终结果确定、杯子放回后调用一次即可。

配置：`configs/result_feedback.json`。

| 参数 | 含义 |
| --- | --- |
| `speed_percent` | 机械臂普通运动速度百分比，默认 30 |
| `finger_duration_s` | 手指动作时间，定时模式默认 0.5 秒；同一动作时间越短越快，范围 0.5–2.55 秒；最大速度模式下此值不控制速度 |
| `gestures.<名称>.joints_deg` | J1–J7 的绝对角度，单位度 |
| `gestures.<名称>.hand_0_100` | 拇指尖、拇指根、食指、中指、无名指、小指六路目标 |

`--gestures <文件>` 可指定手势配置；`--config <文件>` 可指定系统配置，默认使用 `cup_grasp_demo/calibration_debug/green_open_cup/stereo_config.json`。`--session <目录>` 指定记录目录，默认 `cup_grasp_demo/datasets/result_feedback`。每次执行保留请求、反馈与收据；程序退出码为 0 表示指令流程完成，2 表示失败，130 表示用户中断。手指完成按指令时间计，收据不代表已实测手指姿态。

### 增删动作、单动作速度与执行时延

动作名称直接读取 `configs/result_feedback.json` 的 `gestures`，无须修改 Python。复制一个动作并改名即可新增，删除对应键即可删除。动作名使用英文字母、数字、下划线或连字符；`aliases` 定义别名，删除动作时也应删除或更新指向它的别名。

每个动作可覆盖顶层的 `speed_percent`、`finger_duration_s` 和 `execution`。未填写的参数继承顶层默认值。两个内置动作已显式填写速度与时序，调整它们时请修改各自动作中的参数。

```json
"yeah": {
  "joints_deg": [0.116, -90.329, -20.379, 100.034, -9.853, -5.002, 5.068],
  "hand_0_100": [100, 100, 0, 0, 100, 100],
  "speed_percent": 30,
  "finger_duration_s": 0.5,
  "execution": {"mode": "together", "delay_s": 0.2}
}
```

| `execution.mode` | `delay_s` 的含义 |
| --- | --- |
| `arm_then_hand` | 机械臂动作完成后，间隔指定秒数，再执行手指动作。旧配置缺省模式 |
| `hand_then_arm` | 手指动作时间结束后，间隔指定秒数，再移动机械臂 |
| `together` | 从第一条机械臂运动指令起，延迟指定秒数发送手指动作；0 表示一起启动 |

`delay_s` 范围 0–30 秒。`together` 的时延大于机械臂运动时间时，手指将在机械臂结束后才启动。时延是软件指令调度时间，不保证电机物理运动严格同步；执行收据记录实际调度时差。两者共用一个 SDK 连接和控制线程，不启动第二个 CAN 控制器。

```bash
bash run_feedback.sh --list                 # 列出动作
bash run_feedback.sh yeah                   # 查看合并后的参数
bash run_feedback.sh yeah --execute         # 按配置执行，无额外确认输入
bash run_feedback.sh my_action --execute    # 新增动作的调用方式
```

本脚本不会自动修改控制器加速度上限；机械臂速度仍受控制器及系统配置的速度、加速度预算约束。

### 灵巧手最大速度

当前 yeah、thumbs-up 已配置 `finger_speed_mode: "max"`。每个动作可单独修改，未填写时继承顶层配置；旧配置没有此字段时仍使用 `timed`。

```json
"finger_speed_mode": "max",
"finger_duration_s": 0.5,
"finger_max_wait_s": 0.65
```

- `max`：发送目标位置，再发送六路时间全 0，请求控制器允许的最大速度；不修改电机固件上限。
- `timed`：使用 `finger_duration_s` 指定到位时间，恢复原来的定时控制。
- `finger_max_wait_s`：最大速度模式下，下发后保留的观察时间，范围 0.65–5 秒。它不降低运动速度，也不代表实测到位；并行动作会与机械臂运动重叠。顺序模式从该时间结束后开始计算 `delay_s`。

[厂家控制模式说明](https://www.brainco-hz.com/docs/revolimb-hand/en/revo2/parameters.html)明确规定位置＋时间模式的时间为 0 时使用最大速度；产品屈伸时间指标为 ≤0.65 秒。实际时间受行程、负载和固件限制影响，尚未实测当前设备的最大速度。机械臂速度仍为各动作的 `speed_percent`，本设置只影响灵巧手。

### 平局：手指往返动作

```bash
cd /home/test2/dice_demo
bash run_feedback.sh tie             # 预览
bash run_feedback.sh tie --execute   # 执行；draw 为同义别名
```

`gestures.tie` 使用示教关节角 `[0.122, -80.481, -90.400, 110.059, 155.191, -4.993, 5.265]` 度，机械臂速度 50%，臂手同时启动。机械臂只移动到该姿态并保持；手指最大速度往返，最后全张开。不会自动回 HOME。

```json
"hand_0_100": [0, 0, 0, 0, 0, 0],
"hand_sequence": {
  "poses": [[0, 0, 0, 0, 0, 0], [0, 0, 40, 40, 40, 40]],
  "cycles": 3,
  "interval_s": 0.65
}
```

`cycles` 是往返次数（1–20）；一次为 A→B→A。默认发送 A、B、A、B、A、B、A，共 3 次往返。`interval_s` 是相邻手势指令的最小间隔，不是手指限速；最大速度模式下不低于 `finger_max_wait_s`。默认最后一条指令也观察 0.65 秒，手部序列约 4.55 秒，实际可能受调度延迟影响。晚到的指令不会集中补发。

动作中省略 `hand_sequence` 则保持原来的单一手势行为。仍可使用 `arm_then_hand` / `hand_then_arm`；顺序模式可能增加反馈读取开销。

## FAST 启动耗时

FAST 启动性能通过 `green_cup.fast_parallel_startup` 开关对比。设为 `false` 恢复按阶段初始化；设为 `true` 让只读资源初始化与模块加载并行。比较 `green_pipeline_state.json` 的 `startup_to_capture_s` 时，应同时保留命令总耗时，避免只比较阶段数字。相机预热帧数、手指动作时间和识别质量阈值不随此开关变化。

## 摇晃指令下发频率

`configs/joint_shake.json` 的 `command_rate_hz=200` 表示每 5 ms 更新一次七轴 `move_js()` 目标，不是每秒摇晃 200 次。开发入口的 `calibration_debug/joint_test_config.json` 使用同一参数。支持 20–200 Hz；省略或设为 `null` 保留原来等待新反馈后发送的循环。

200 Hz 模式使用独立只读反馈线程，发送线程按单调时钟调度；反馈过期、故障及运动约束检查仍有效。迟到时跳过错过的时隙，不连续补发积压目标。Python/Linux 调度与 CAN 发送仍可能有抖动，不能把配置值当作实测频率。

摇晃 `actual.json` 中的 `command_stream` 记录 `requested_hz`、`achieved_hz`、`max_interval_ms`、`skipped_slots` 和 `max_lateness_ms`；频率基于 SDK 调用完成时间，不是 CAN 总线抓包时间。改配置后重新运行 Pipeline，独立关节测试需重新 plan。
