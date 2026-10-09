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
- [参数配置参考](docs/CONFIGURATION.md)
- [运行中修改参数（热加载）](docs/HOT_RELOAD.md)
- [第三方依赖与分发范围](THIRD_PARTY.md)

## 1. 获取代码

```bash
git clone https://github.com/wangannyi/dice_demo.git
cd dice_demo
```

## 2. 安装和检查环境

K3 上一键安装系统依赖：

```bash
sudo bash scripts/bootstrap_k3.sh
```

安装内容、依赖锁定及解释器选择见[运行环境文档](docs/ENVIRONMENT.md)。

```bash
source scripts/env.sh --system
bash scripts/check_environment.sh
```

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

分阶段停止与单项测试见[调试指南](docs/DEBUG.md#2-分阶段运行)；常驻进程协议见[接入文档](docs/INTEGRATION.md)。

## 5. 配置入口

| 文件 | 用途 |
| --- | --- |
| [`configs/green_cup.json`](configs/green_cup.json) | 主配置：硬件、标定路径、运动速度、放杯和识别参数 |
| [`vision/strategy/green_cup.json`](vision/strategy/green_cup.json) | 抓取策略：TCP、腕部姿态、抬杯高度和手指目标 |
| [`vision/camera.json`](vision/camera.json) | 相机分辨率、帧率和裁剪 |
| [`configs/actions/joint_shake.json`](configs/actions/joint_shake.json) | 摇晃动作 |
| [`configs/actions/gestures/`](configs/actions/gestures/) | 猜拳和胜负反馈动作 |

字段含义、优先级和放杯模式见[参数配置参考](docs/CONFIGURATION.md)；修改后何时生效见[热加载说明](docs/HOT_RELOAD.md)。

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

## 9. 标定与外参恢复

在仓库根目录执行以下命令，参数统一在 [`configs/calibration_workflow.json`](configs/calibration_workflow.json) 中配置。

| 场景 | 命令 |
| --- | --- |
| 首次标定：人工示教采样并求解 | `bash calibrate.sh first` |
| 自动标定：沿已有示教路线采样并求解 | `bash calibrate.sh auto --execute` |
| 相机移动后：通过已登记的固定板恢复外参 | `bash calibrate.sh restore --execute` |

复用内置 20 姿态轨迹时，先确认满足[标定指南](docs/CALIBRATION.md)中的安装条件，再运行 `bash calibrate.sh plan` 生成本机计划。

得到标定结果后，按指南完成现场准备，再运行 `bash calibrate.sh apply`，自动备份原结果、应用新结果并登记桌面。完整流程与配置说明见[标定指南](docs/CALIBRATION.md)。
