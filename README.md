# K3 机械臂抓杯摇骰 Pipeline

本项目在 SpacemiT K3 上控制 NERO 七轴机械臂、右 Revo2 灵巧手和 RealSense D435i，完成绿杯定位、抓取、抬杯、摇晃、放杯和归位。

```text
HOME → CAPTURE → PLAN → APPROACH → GRIP → LIFT → SHAKE → LOWER → OPEN → RETURN_HOME
```

## 文档入口

- [运行环境与安装](docs/ENVIRONMENT.md)
- [首次标定、自动重采和相机移动后的恢复](docs/CALIBRATION.md)
- [浏览器手眼标定工作台](docs/CALIBRATION_WEB.md)
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

### Web 标定工作台

在 K3 上需要应用标定结果的 `dice_demo` 仓库根目录启动，电脑通过浏览器访问：

```bash
# 首次使用时设置工程师 PIN；已有 PIN 时跳过。
bash calibrate_web.sh --set-pin
bash scripts/calibration_web_service.sh start
bash scripts/calibration_web_service.sh status
```

打开 `http://<K3地址>:8091`，先正常退出占用相机、机械臂的程序，再点击“连接并检查设备”。准备完成后，按页面提示进入下一步。独立工具只更新页面底部显示的配置路径，不会同步到另一份 `dice_demo`。

| 入口 | 适用情况 | 操作顺序 |
| --- | --- | --- |
| 首次安装 | 新安装，或基座、末端板安装关系改变 | 设备检查 → 人工示教 → 画可见范围 → 生成路线 → 自动采样 → 登记固定板 → 应用并登记桌面 |
| 重新标定 | 安装关系不变，复用已有示教路线 | 设备检查 → 生成路线 → 自动采样 → 登记固定板 → 应用并登记桌面 |
| 相机位置变了 | 仅移动相机，已登记的桌面固定板和基座均未移动 | 设备检查 → 恢复相机位置 → 应用并登记桌面 |

人工示教会保存关节路线和采样数据，生成路线后可用于下次自动标定；Web 优先使用已保存的新示教。修改示教、对应标定或可见范围后需重新生成路线。绿色框表示图像中的采样可见范围，不能代替现场检查 HOME 往返路径和机械臂活动空间。

自动标定启动及 HOME 姿态无需看见标定板，到达采样姿态后才检查。采样时固定末端板并遮挡桌面同型板。Web 求解后自动回 HOME，并向灵巧手发送六路归零指令；不等待手指位置反馈。确认机械臂到位、静止后，再露出固定板进行登记。

### 检查和应用结果

“标定结果”显示质量、误差和有效样本数；“每组标定误差”和“标定检测图片”默认收起，可展开查看逐组误差、原图、检测标注及下载数据。全样本拟合反映已有数据的一致性，留出验证检查未参与该次拟合的样本；两者都不能直接等同于实际抓取误差。

当前验收门槛为 **15 mm / 5°**，在 [`configs/calibration_workflow.json`](configs/calibration_workflow.json) 的 `quality_thresholds` 中修改，修改后重启服务。超限结果用于关键步骤时，需要工程师 PIN、确认误差并填写原因；同一结果和门槛的批准可复用，质量仍标记为未达标。

求解和外参恢复均不会自动应用结果。“应用并登记桌面”先备份原配置，再应用标定、重新采集并绑定桌面。操作前拆下末端板，确认机械臂在 HOME、红布桌面露出；桌面登记失败可重试。相机移动后必须重新登记桌面，不能继续使用旧桌面数据。固定板或基座也移动、相机更换或内参改变时，需重新做手眼标定。

采样、逐帧检测图片、结果、备份和操作记录保存在 `calibration/datasets/web/`。作业期间保持启动页面连接；关闭页面或心跳中断超过 10 秒会请求停止，重连后检查设备，不自动续跑。停止服务使用 `bash scripts/calibration_web_service.sh stop`。部署、数据格式和停止行为见 [Web 工具说明](docs/CALIBRATION_WEB.md)。

### 命令行入口

命令在仓库根目录执行，参数统一配置在 [`configs/calibration_workflow.json`](configs/calibration_workflow.json)。CLI 与 Web 的状态目录独立，CLI 的 HOME 只控制机械臂，自动采样后不自动回 HOME。

| 目的 | 命令 |
| --- | --- |
| 人工示教并求解 | `bash calibrate.sh first` |
| 画可见范围 / 生成自动路线 | `bash calibrate.sh window` / `bash calibrate.sh plan` |
| 沿已有路线自动采样并求解 | `bash calibrate.sh auto --execute` |
| 登记固定板 / 相机移动后恢复外参 | `bash calibrate.sh register` / `bash calibrate.sh restore --execute` |
| 应用结果并登记桌面 / 重试桌面登记 | `bash calibrate.sh apply` / `bash calibrate.sh table` |
| 用已有样本重新求解 / 查看状态 | `bash calibrate.sh solve` / `bash calibrate.sh status` |

任意操作加 `--dry-run` 可预览命令，不连接设备。复用仓库内置 20 姿态轨迹前，须核对原安装条件，并在本机重新生成计划；新场地不能直接沿用。完整命令顺序、路径选择及固定板恢复条件见[标定指南](docs/CALIBRATION.md)。
