# K3 机械臂抓杯摇骰 Pipeline

本项目在 SpacemiT K3 上控制 NERO 七轴机械臂、右 Revo2 灵巧手和 RealSense D435i，完成绿杯定位、抓取、抬杯、摇晃、放杯和归位。

```text
HOME → CAPTURE → PLAN → APPROACH → GRIP → LIFT → SHAKE → LOWER → OPEN → RETURN_HOME
```

## 文档入口

- [运行环境与安装](docs/ENVIRONMENT.md)
- [首次标定、自动重采和相机移动后的恢复](docs/CALIBRATION.md)
- [分阶段调试与单项测试](docs/DEBUG.md)
- [上层应用接入接口](docs/INTEGRATION.md)
- [第三方依赖与分发范围](THIRD_PARTY.md)

## 1. 获取代码

```bash
git clone https://github.com/wangannyi/dice_demo.git
cd dice_demo
```

创建交付包：

```bash
python3 scripts/package_release.py --output "dist/release_$(date +%Y%m%d_%H%M%S)"
```

交付包包含源码、配置、模型和标定工具，不包含运行记录。默认交付包要求接收方重新标定。只有复现同一套固定设备时才使用 `--site-active` 保留现场标定状态。

## 2. 安装和检查环境

K3 上一键安装系统依赖：

```bash
sudo bash scripts/bootstrap_k3.sh
```

脚本通过 Bianbu 安装 NumPy、SciPy、OpenCV、ONNX Runtime、python-can 等系统包，并安装仓库附带且经过校验的 K3/CPython 3.14 RealSense wheel。项目默认使用系统 `python3`；NERO SDK 固定在 `third_party/pyAgxArm/`。不要复制其他机器的虚拟环境。

```bash
source scripts/env.sh
bash scripts/check_environment.sh
```

脚本不依赖仓库所在的绝对路径，也不会搜索用户主目录中的虚拟环境。需要使用非默认解释器时，显式设置 `DICE_PYTHON`。

完整依赖和环境变量见[运行环境文档](docs/ENVIRONMENT.md)。

## 3. 硬件准备

1. 固定机械臂基座、D435i、桌面和固定标定板。
2. 解除急停并使能七个关节。
3. 在 WEB 页面选择 Revo2、打开灵巧手使能并开启 CAN 推送。
4. 确认相机和 CAN：

```bash
lsusb
lsusb -t
ip -details link show can0
```

`can0` 未启动时执行：

```bash
sudo ip link set can0 up type can bitrate 1000000
```

`Device or resource busy` 且接口显示 `UP` 时无需重复设置。运行期间关闭占用相机的程序，并确保没有第二个机械臂控制进程。

新安装、相机移动或基座变化后，先按[标定文档](docs/CALIBRATION.md)完成手眼标定和桌面登记。

## 4. 运行 Pipeline

在仓库根目录执行：

```bash
# 预览流程，不连接硬件
bash run.sh fast

# 连续完成抓杯、摇晃、放杯和归位
bash run.sh fast --execute

# 常驻进程，由上层逐阶段调度
bash run.sh control --execute
```

默认配置和会话目录：

```text
configs/green_cup.json
cup_grasp_demo/datasets/green_current
```

可通过环境变量覆盖：

```bash
DICE_CONFIG="$PWD/configs/green_cup.json" \
DICE_RUN="$PWD/cup_grasp_demo/datasets/job_001" \
bash run.sh fast --execute
```

### 运行模式

| 模式 | 用途 |
| --- | --- |
| `fast` | 一次连续执行到目标阶段，复用 SDK、相机和模型 |
| `control` | 进程常驻，通过逐行 JSON 命令推进阶段 |

`fast` 可用以下停止点：

```bash
bash run.sh fast --until ready --execute  # 到抓取位置
bash run.sh fast --until grip --execute   # 闭手后停止
bash run.sh fast --until shake --execute  # 摇晃后停止，可能仍持杯
bash run.sh fast --until place --execute  # 放杯并返回 HOME
```

这些命令每次都从 HOME 开始，不支持跨进程续跑。需要停在某阶段并保留连接时使用 `control`，协议见[接入文档](docs/INTEGRATION.md)。

## 5. 主要配置

主配置为 [`configs/green_cup.json`](configs/green_cup.json)。常用字段如下：

| 配置 | 含义 |
| --- | --- |
| `serial`、`channel` | RealSense 序列号和 CAN 接口 |
| `calibration` | 当前安装的手眼标定结果 |
| `home` | HOME 七轴姿态 |
| `green_cup.home_table_scene` | 已登记的桌面平面 |
| `green_cup.installation_requires_calibration` | `true` 时禁止真机 Pipeline，需先完成标定和桌面登记 |
| `green_cup.strategy_file` | 抓取策略文件；其中包含 TCP、接触点、腕部、抬杯和手指参数 |
| `green_cup.fast_speed_percent` | FAST 普通运动速度百分比 |
| `green_cup.fast_phase_speed_percent` | 指定阶段的速度覆盖值 |
| `green_cup.place_offset_base_mm` | 放杯目标相对抓取位置的基座坐标系 `[X, Y, Z]` 补偿，单位 mm |
| `green_cup.perception` | 绿杯模型、尺寸、杯沿和推理后端 |
| `vision/camera.json` | 彩色、深度、双目分辨率、帧率和裁剪 |
| `green_cup.joint_test_config` | 摇晃动作配置文件 |

六路手指顺序为：拇指尖、拇指根、食指、中指、无名指、小指。TCP 偏移使用法兰坐标系，不是图像坐标系。

### 相机档位

```bash
python3 scripts/set_camera_profile.py usb2
python3 scripts/set_camera_profile.py usb3
python3 scripts/set_camera_profile.py usb2 --dry-run
```

默认 USB 2.0 档位为 1280×720、6 FPS；USB 3.0 档位为 1280×720、15 FPS。命令会同步抓杯与标定配置。改变彩色分辨率或裁剪后必须重新标定；改变深度分辨率后必须重新验证深度对齐和杯位。

### 摇晃动作

[`configs/actions/joint_shake.json`](configs/actions/joint_shake.json) 配置参与关节、幅度、速度、加速度、周期和目标更新频率。`command_rate_hz` 是七轴位置目标的发送频率，不是杯子的往返频率。实际频率受行程、轨迹和控制器限制。

## 6. 骰子反馈与猜拳动作

```bash
# 交互常驻：初始化一次 SDK/CAN，动作完成后返回菜单，q 退出
bash run_feedback.sh --execute

# 非交互调用，适合上层程序
bash run_feedback.sh win --execute
bash run_feedback.sh lose --execute
bash run_feedback.sh draw --execute

# 猜拳
bash run_feedback.sh rock --execute
bash run_feedback.sh paper --execute
bash run_feedback.sh scissors --execute

# 张手并返回 HOME
bash run_feedback.sh home --execute
```

不带 `--execute` 进入单次交互预览，不连接 CAN 或发送动作。交互执行模式只在启动时初始化一次 SDK/CAN；此后可连续切换动作。带动作名的命令仍执行一次后退出，供上层程序调用。

| 结果 | 动作 |
| --- | --- |
| `win` | 比 V；臂 100%，手最大速度 |
| `lose` | 点赞；臂 100%，手最大速度 |
| `draw` | 平局往返手势；臂 100%，手型每 0.5 秒切换 |
| `rock` | 石头；四指开始闭合后 0.1 秒拇指即跟进，两段均使用最大速度 |
| `paper` | 布；六路手指全张开 |
| `scissors` | 剪刀；食指和中指张开 |
| `home` | 六路手指张开，机械臂返回保存的 HOME 关节姿态 |

动作定义在 [`configs/actions/gestures/`](configs/actions/gestures/)。动作执行后保持姿态，不自动回 HOME。添加动作、速度和臂手时序配置见[调试文档](docs/DEBUG.md#6-反馈动作)。

## 7. 输出文件

每个会话目录包含：

| 文件或目录 | 内容 |
| --- | --- |
| `green_pipeline_state.json` | 当前状态、阶段耗时和错误 |
| `green_grasp_plan.json` | 抓取目标和规划结果 |
| `runs/` | 每次执行的请求、实际结果和日志 |

程序返回成功表示动作流程完成，不表示视觉或力觉已经确认抓牢，也不表示骰子点数一定改变。失败后先检查实物状态和本次日志，再决定是否张手、放杯或归位。

## 8. 项目结构

```text
run.sh                         Pipeline 入口
run_feedback.sh                胜负反馈入口
configs/                       系统、HOME、摇晃和手势配置
calibration/                   手眼标定、内置自动轨迹和固定板工具
cup_grasp_demo/flow/           Pipeline 状态机、规划和执行
vision/                        相机、推理和几何计算
nero_revo2_control/            NERO/Revo2 控制与模型
scripts/                       安装、环境、桌面登记和交付工具
docs/                          标定、调试、环境和集成文档
tests/                         回归测试
```

## 简化标定入口

在仓库根目录运行 `bash calibrate.sh first`（首次人工示教）、`bash calibrate.sh auto --execute`（自动标定）或 `bash calibrate.sh restore --execute`（固定板恢复）。参数统一编辑 `configs/calibration_workflow.json`。`bash calibrate.sh apply` 自动备份、应用结果并登记桌面；完整步骤见 [标定指南](docs/CALIBRATION.md)。 内置 20 姿态轨迹首次使用前运行 `bash calibrate.sh plan`。
