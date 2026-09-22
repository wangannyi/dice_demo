# K3 机械臂抓杯摇骰 Pipeline

在 SpacemiT K3 上，通过 NERO 七轴机械臂、右 Revo2 灵巧手和 RealSense D435i，完成绿色开口杯的定位、抓取、抬起、摇晃和放回。杯口朝上，工作区为红色桌布。

```text
HOME → CAPTURE → PLAN → APPROACH → GRIP → LIFT → SHAKE → LOWER → OPEN → RETURN_HOME
归位     定位      规划     靠近       闭手    抬杯     摇晃      放下    张手       归位
```

- [K3 系统依赖与 Python 环境实测清单](docs/ENVIRONMENT.md)
- [分步调试与单项测试](docs/DEBUG.md)
- [首次人工标定、画框与自动重采、相机移动后的校准](docs/CALIBRATION.md)
- [上层应用接入接口](docs/INTEGRATION.md)
- [第三方依赖与分发范围](THIRD_PARTY.md)

## 1. 获取代码

源码包包含程序、配置、检测模型、机械臂几何模型及一套参考标定结果，不包含虚拟环境、历史录像、运行日志或 SDK 二进制库。

```bash
sha256sum -c SHA256SUMS
tar -xzf dice_demo-source.tar.gz
cd dice_demo
```

也可直接拉取仓库：

```bash
git clone https://github.com/wangannyi/dice_demo.git
cd dice_demo
```

构建新的源码包：

```bash
python3 scripts/package_release.py --output "dist/release_$(date +%Y%m%d_%H%M%S)"
```

输出包含 `dice_demo/` 源码目录、`dice_demo-source.tar.gz`、`SHA256SUMS`。目录内 `MANIFEST.sha256.json` 记录逐文件哈希。打包不会提交、上传或操作机械臂。

仓库 `main` 保存当前 K3 安装的配置快照；其中相机标定、板尺寸和动作幅度只对应这套固定现场。向新设备交付时使用上面的默认打包命令：它将相机标定路径指向包内参考文件，并把 `installation_requires_calibration` 设为 `true`。仅用于复现当前 K3 文件时，显式加 `--site-active`，保留现场配置原值；不要将此包直接用于另一套安装。

## 2. 安装运行环境

### 2.1 软件依赖

当前 K3 的系统版本、Python 包版本、加载路径及厂商运行库见[环境依赖清单](docs/ENVIRONMENT.md)（2026-09-22 核对）。

需要 Linux SocketCAN、Python、NumPy、SciPy、带 ArUco 的 OpenCV、RealSense Python SDK、ONNX Runtime、python-can 和与 NERO 固件匹配的 pyAgxArm。ROS/MoveIt 和 MediaPipe 不是当前流程的运行依赖。

K3 为 RISC-V。优先使用板卡提供的兼容库或已经验证的环境；不能复制 PC 的 x86 虚拟环境，也不能保证 PyPI 为 RISC-V 提供所有轮子。`requirements-vision.txt`、`requirements-sdk.txt` 是依赖清单，不是跨架构安装成功的承诺。

已有 K3 环境默认使用：

| 变量 | 默认位置 | 用途 |
| --- | --- | --- |
| `DICE_VISION_PYTHON` | `$HOME/.venv-grasp/bin/python` | 相机、识别、几何、规划 |
| `DICE_SDK_PYTHON` | `$HOME/agilex-api-test/venv/bin/python` | CAN 和灵巧手执行器 |
| `NERO_SDK_DIR` | `$HOME/agilex-api-test/pyAgxArm` | 已验证 SDK 源码根目录 |
| `CALIB_PYTHON` | 与视觉解释器相同 | 标定 |

新环境先安装本架构可用的依赖。具备对应轮子或源码构建环境时：

```bash
python3 -m venv --system-site-packages .venv-vision
python3 -m venv --system-site-packages .venv-sdk
.venv-vision/bin/python -m pip install -r requirements-vision.txt
.venv-sdk/bin/python -m pip install -r requirements-sdk.txt
# 将已验证的 pyAgxArm 源码放到指定目录；目录内应有 pyAgxArm/ 包。
export NERO_SDK_DIR=/实际路径/pyAgxArm
export DICE_VISION_PYTHON="$PWD/.venv-vision/bin/python"
export DICE_SDK_PYTHON="$PWD/.venv-sdk/bin/python"
source scripts/env.sh
./scripts/check_environment.sh
```

检查命令只导入依赖并验证配置和网格，不打开相机或 CAN。SDK 与视觉环境可以共用解释器，但必须先验证所有依赖。`green_cup.perception.ort_package_dir` 指向 ONNX Runtime 的备用安装目录；当前 K3 为 `/usr/lib/python3.14/dist-packages`，其他环境应按安装位置调整。需要 X11 预览时，在 PC 使用 `ssh -X 用户@K3地址`，K3 安装 `xauth` 并确认 `DISPLAY` 已设置。

### 2.2 硬件准备

1. 固定机械臂基座、相机和桌面板，连接灵巧手。
2. 执行 `lsusb` 和 `lsusb -t`，确认 D435i 已枚举且链路为 `480M`（USB 2.0）或 `5000M`（USB 3.0）。抓杯检测及红布标定板默认均为 1280×720、6 FPS，适用于已验证的 USB 2.0 采集；需要 1280×720、15 FPS 的彩色/深度/双目组合时切换 USB 3.0 配置。
3. 执行 `ip -details link show can0`，确认 CAN 为 UP、1 Mbps。尚未启动时执行：

```bash
sudo ip link set can0 up type can bitrate 1000000
```

已 UP 时不要重复设置 bitrate；`Device or resource busy` 不等于 CAN 已故障。执行 Pipeline 的 `--execute` 入口会检查 `can0`：已 UP 时直接继续；USB-CAN 重插后若为 DOWN，会尝试恢复到 1 Mbps，终端可能要求输入 sudo 密码。非交互运行且无 sudo 权限时会立即打印上面的手工命令并退出。该检查只恢复 Linux 接口，不代替控制器 CAN 推送或灵巧手使能。确认急停解除、机械臂七轴使能、WEB 灵巧手页面使能及 CAN 推送开启。程序支持从 WEB 切入 CAN。运行期间关闭占用相机的 ffplay，不使用第二个机械臂控制程序。

新安装先完成[标定](docs/CALIBRATION.md)。默认发行包将 `installation_requires_calibration` 设为 `true`，避免将当前 K3 标定当作新现场的有效标定。

## 3. 一条命令运行

从仓库根目录执行。默认配置为 `configs/green_cup.json`，运行结果保存在 `cup_grasp_demo/datasets/green_current`。

```bash
# 不访问硬件，只打印流程
./run.sh fast

# 分步运行；每阶段 Enter 继续、q 停止
./run.sh step --show --execute

# 上层应用常驻调度；按 JSON 指令推进到指定阶段
./run.sh control --execute

# 自动连续运行，保留非 FAST 的执行与诊断路径
./run.sh auto --execute

# 快速连续运行，复用 SDK、相机和模型
./run.sh fast --execute
```

| 模式 | 阶段确认 | 诊断 | 运动设置 |
| --- | --- | --- | --- |
| `step` | 每步 Enter | `--show` 可显示图像和 TCP | `speed_percent` |
| `control` | 上层 JSON 指令推进，阶段间常驻等待 | JSON 事件与状态文件；详细日志在 stderr | 与 STEP 相同的阶段运动配置 |
| `auto` | 连续执行 | 保留常规图像及记录；不弹 STEP 窗口 | `speed_percent` |
| `fast` | 连续执行 | 精简图像；保存状态、计划、收据和错误 | FAST 专用参数 |

默认运行到放杯并返回 HOME。可加 `--until ready`、`--until grip`、`--until shake` 分别停在靠近、闭手、摇完；停在后两者时可能仍持杯。没有 `--execute` 不会运动。

切换配置或输出位置：

```bash
DICE_CONFIG="$PWD/configs/green_cup.json" \
DICE_RUN="$PWD/cup_grasp_demo/datasets/green_current" \
./run.sh fast --execute
```

不要在执行中改配置。HOME 会先发张手命令；故障后若仍持杯，不要直接重新启动流程。绿色流程目前不支持跨进程 `--resume`。

## 4. 重要配置

数值来自交付时 K3 配置快照，不使用聊天中的旧值。修改新入口使用的 `configs/green_cup.json`，不要同时维护旧入口配置。

配置以文件实际值为准，修改后重新启动流程。普通到位精度使用 `record` 记录策略；通信故障、无有效目标、关节越界及碰撞等错误仍可能停止流程。失败状态和已完成阶段写入会话文件，见[接入接口](docs/INTEGRATION.md)。阶段耗时包括连接、规划、动作和反馈，不是单纯的电机运动时间。

FAST 复用 SDK/CAN、相机和模型，复用可用的预计算轨迹；只在定位阶段识别杯子，减少诊断图片和固定等待。STEP 在首次提示前连接 SDK、启动相机并预热、加载模型，随后在同一次命令内保持连接，逐阶段确认和预览使用新采图。`control` 为上层程序提供 JSON 指令：可运行到指定阶段停住，保持进程与设备连接，等下一条指令再继续；完整协议见[接入接口](docs/INTEGRATION.md#常驻阶段控制供上层集成)。各模式均不能把指令发送成功解释成已抓牢。

绿色杯流程使用标定阶段保存的桌面平面；CAPTURE 只识别本次杯口，不再逐帧拟合桌面。首次标定及相机重新校准后的桌面登记命令见[标定文档](docs/CALIBRATION.md#6-应用标定并更新桌面)。桌面或基座改变后必须重做桌面登记。

| 参数 | 交付值 | 含义 |
| --- | --- | --- |
| `serial` / `channel` | `346222071954` / `can0` | 相机与 CAN |
| `calibration` | 参考标定文件路径 | 彩色相机到基座变换；新安装须更换 |
| `home` | HOME JSON 路径 | 七轴 HOME 角度，度 |
| `speed_percent` | 20 | STEP/AUTO 普通运动百分比 |
| `green_cup.fast_speed_percent` | 30 | FAST 普通运动百分比 |
| `green_cup.fast_phase_speed_percent` | approach、return_home 均 60 | 两段单独速度百分比 |
| `green_cup.contact_offset_base_mm` | `[0,0,30]` | 杯口圆心在基座坐标系中的 TCP 目标偏移，mm |
| `green_cup.tcp_offset_flange_mm` | `[30,15,0]` | 在 TCP 文件基础上、沿法兰坐标轴追加的偏移，mm |
| `green_cup.wrist_reference_deg` | `[15,-13,5]` | J5/J6/J7 的 IK 偏好，不是固定锁定 |
| `green_cup.lift_mm` | 50 | 抬杯高度，mm |
| `green_cup.open_targets_0_100` | `[0,0,0,0,0,0]` | 张手目标 |
| `green_cup.grip_targets_0_100` | `[0,100,40,40,40,100]` | 闭手目标 |
| `green_cup.finger_duration_s` | 1 | 常规手指动作时间，秒 |
| `green_cup.fast_finger_duration_s` | 0.25 | FAST 手指指令动作时间，秒；不代表实测抓牢 |
| `green_cup.fast_minimize_lift_travel` | true | FAST 抬杯重新分配七轴位移，保持 TCP 终点和朝向；优化失败回退原解，仍检查持杯路径 |
| `green_cup.fast_motion_profile` | trapezoid | FAST 普通关节运动使用限速、限加速度的梯形速度曲线；设为 quintic 恢复原曲线。SHAKE 不受此项影响 |
| `green_cup.fast_dogbox_ik` | true | FAST 使用 dogbox 求解抓取 IK；仍验证位置、朝向与路径，无合格解时回退原求解器 |
| `green_cup.fast_parallel_startup` | `true` | FAST 执行时，SDK 连接、相机预热及模型加载与 CLI 模块加载并行；初始化不发送运动或手指指令 |
| `green_cup.persistent_runtime` | `true` | STEP、CONTROL 和 FAST 在同一进程内保持 SDK 与相机连接；上层集成时同一时刻只能有一个任务占用设备 |
| `green_cup.table_plane_source` | `calibrated` | 从 `home_table_scene` 读取标定阶段保存的基座桌面平面；`live_depth` 恢复每次 CAPTURE 拟合桌面 |
| `green_cup.perception.height_mode` | `fixed` | `fixed` 已知杯高；`measured` 双目测高 |
| `green_cup.perception.fixed_height_mm` | 65 | 固定模式杯高，mm |
| `green_cup.perception.inference_provider` | `spacemit` | K3 AI 后端；`cpu` 使用普通 CPU 后端 |
| `green_cup.perception.inference_threads` | 2 | AI 后端计算线程数；CPU 后端时为 CPU 推理线程数 |
| `green_cup.perception.inference_cpu_ids` | `[8,9]` | 两个 A100 AI 核；CPU 后端设为 `[]` |
| `green_cup.fast_camera_warmup_frames` | 5 | STEP/FAST 常驻相机启动时的预热帧数 |
| `green_cup.fast_camera_fresh_discard_frames` | 0 | FAST 正式采集前额外丢帧数；仍清理旧队列并要求 RGB/深度帧号推进 |
| `green_cup.camera` | RGB/深度 1280×720、6 FPS | RGB 裁剪 `[220,0,960,720]`；程序同步修正内参。USB 2.0 下已完成采集与绿杯定位验证；完整运动流程尚需实测 |
| `green_cup.joint_test_config` | `configs/joint_shake.json` | 摇晃配置 |
| `green_cup.rtsp` | `{enabled, host, port, path}` | 摄像头 RTSP 推流开关与目的地；见下文 |

### 相机配置

抓杯检测与手眼、固定板标定分别读取自己的配置文件；以下命令会同步修改三份绿杯配置及两份红布标定板配置。在 PC 或 K3 的项目根目录执行：

```bash
python scripts/set_camera_profile.py usb2  # 默认：RGB/深度/双目 1280×720、6 FPS
python scripts/set_camera_profile.py usb3  # RGB/深度/双目 1280×720、15 FPS
python scripts/set_camera_profile.py usb2 --dry-run  # 只预览，不写文件
```

可用 `--fps`、`--color-resolution WIDTHxHEIGHT`、`--depth-resolution WIDTHxHEIGHT`、`--crop X,Y,W,H` 指定采集参数。已实测的 USB 2.0 四路同步采集组合为 1280×720@6；配置工具也允许彩色/深度同为 640×480@6/15，切换后仍须在目标接口上实拍确认。USB 3.0 默认 1280×720@15。切换帧率但保持同一相机、分辨率和裁剪时，固定板注册/恢复可复用空间内参。首次示教数据不会被改写；自动重复标定如需改用 6 FPS，在 `auto_collect.py run` 命令增加 `--fps 6`，见[标定文档](docs/CALIBRATION.md#34-自动重复采样)。

变更彩色分辨率或裁剪时，还必须传入按**新画面**确定的 `--hand-roi X1,Y1,X2,Y2` 和 `--reference-roi X1,Y1,X2,Y2`。脚本会将 `installation_requires_calibration` 设为 `true`；此时必须重新标定，不能直接运行抓杯 Pipeline。深度分辨率变化也需要重新验证杯位和深度对齐。设置后先用 `lsusb -t` 核实实际 USB 链路，再用不驱动机械臂的 `green-detect` 验证采集与定位；PC、K3 的配置文件需同步。

杯沿质量参数位于 `green_cup.perception.stereo_rim`：`min_edge_support=0.85` 要求每路图像至少 85% 的采样杯沿点距离观测边缘小于 `edge_distance_px=2.0` 像素。平均边缘误差上限仍为 1.0 px，单路平均上限仍为 1.2 px；该比例不是 YOLO 置信度。

### 摄像头 RTSP 推流

`green_cup.rtsp` 控制常驻模式下的摄像头推流（`enabled`、`host`、`port`、`path`，缺省 `false` 不推流）。开启时，相机读帧线程持续把裁剪后的彩色画面（与识别同一画面、纯原图不叠加识别结果）经 `spacemith264enc` 硬编与 `rtspclientsink` 发布到本机 MediaMTX：拉流地址 `rtsp://<板子IP>:8554/dice/seg`，浏览器 WebRTC 预览 `http://<板子IP>:8889/dice/seg`。推流子进程崩溃或 MediaMTX 未启动时按 5 秒退避自动重启，采集、识别与运动不受影响；`enabled: false` 时相机保持原有按需采集行为，无额外读帧开销。

六路手指顺序为：拇指尖、拇指根、食指、中指、无名指、小指。指令完成不等于已测量确认抓牢。TCP 偏移属于法兰坐标系，不能直接按图像左右方向修改。

FAST 的 HOME/CAPTURE 阶段耗时不包含 CLI 模块加载。`green_pipeline_state.json` 同时记录 `parallel_startup_elapsed_s`（启动到进入状态机）与 `startup_to_capture_s`（启动到定位完成），用后者比较整体启动性能。已在 HOME 时可将定位与张手重叠；不在 HOME 时仍先完成归位再采集正式图像。STEP 在第一条阶段提示前完成相机预热和 SDK 连接；AUTO 和不带 `--execute` 的预览不提前打开设备。

`configs/joint_shake.json`：

| 参数 | 交付值 | 含义 |
| --- | --- | --- |
| `joints` | `[1,4,5,6,7]` | 同时摇晃的关节编号 |
| `amplitude_deg` | `[5,5,5,5,5]` | 当前 K3 主流程各关节单侧幅度；负号表示反向 |
| `velocity_deg_s` | `[170,170,170,200,200]` | 各关节速度预算，°/s |
| `acceleration_deg_s2` | 各 286.4788975654116 | 各关节加速度预算，°/s²，等于 5 rad/s² |
| `cycles` | 6 | 完整往返周期，另有渐入和回中心 |
| `phase_delay_deg` | 省略或 `null` | 按 `joints` 顺序设置各轴相位滞后，0～360°；90° 表示晚四分之一周期开始 |
| `controller_speed_percent` | 100 | 摇晃执行速度百分比 |

当前 `stereo_config.json` 开发入口使用 `cup_grasp_demo/calibration_debug/joint_test_config.json`：`joints=[1,4,5,6,7]`、`amplitude_deg=[4,4,-4,4,4]`、`phase_delay_deg=null`、`cycles=6`。负号使 J5 反向运动。顶层 `configs/joint_shake.json` 是主流程配方，当前幅度为各轴 2.5°；两套配方按用途分别调整。关闭相位延迟推荐使用 `null`，增减关节时不必修改该字段；使用列表时必须与 `joints` 一一对应。当前五轴若设为 `[0,0,0,0,90]`，J7 相对 J6 滞后四分之一周期；每轴均从中心静止启动，完成自身周期后回中心，整体时长增加最大相位延迟。pipeline 在 HOME 之前校验摇晃参数，配置错误时不会先移动再报错。修改后重新生成计划，不执行旧计划。

速度预算不是实际到达速度；最终轨迹仍受关节行程及控制器限值约束。配置里的控制器加速度目标不代表每次 pipeline 都写入硬件参数。

AI 推理通过 `SpaceMITExecutionProvider` 创建 CPU 8、9 上的计算线程。Python、相机、控制及未被 AI 后端接管的算子仍在普通 CPU 上运行。不要用 `taskset` 把整个 Pipeline 绑到 AI 核。配置为 `spacemit` 时依赖 `spacemit_ort`；后端初始化失败会报错。回退时同时设置 `inference_provider="cpu"`、`inference_threads=4`、`inference_cpu_ids=[]`，然后重新启动程序。

## 5. 输出与项目结构

`green_pipeline_state.json` 记录状态、每阶段耗时、路径复用和错误；`green_grasp_plan.json` 记录目标及规划耗时；`green_approach_timings.json` 拆分运动和后续准备；`runs/` 保存每次执行的 `request.json`、`actual.json` 和日志。固定 RUN 复用最新结果，但历史 runs 不自动清空。

```text
configs/                    交付使用的主配置、摇晃配置
scripts/                    环境检查、源码打包、桌面参数登记
cup_grasp_demo/              识别、规划、状态机和调试入口
nero_revo2_control/          机械臂及灵巧手控制、运动学
nero_calibration/           手眼标定、逐帧示教轨迹、自动重采、参考板恢复
rgb_hand_tracking/          复用的 SDK 反馈与控制审计模块
agx_arm_ros/.../agx_arm_urdf/ 几何模型，不要求 ROS 运行时
docs/                       调试和标定指南
```

`rgb_hand_tracking` 中保留历史视觉实验源码是为了兼容被复用的底层模块；当前绿色杯 pipeline 不运行 MediaPipe。软件测试不能替代新安装后的实物接触、抓牢和运动通路验收。

## 6. 比大小后的反馈动作

独立脚本：机械臂赢了比 yeah，输了点赞。完成放杯后调用，动作完成保持姿态；不启动相机或 YOLO。

```bash
cd /home/test2/dice_demo
bash run_feedback.sh yeah --execute       # 机械臂赢
bash run_feedback.sh thumbs-up --execute  # 机械臂输
bash run_feedback.sh tie --execute        # 平局：手指往返 3 次
```

去掉 `--execute` 仅预览。`bash run_feedback.sh --list` 查看动作列表。可在 `configs/result_feedback.json` 增删动作，分别设置机械臂速度、手指动作时间、先后/同时执行与启动时延；[参数与调试说明](docs/DEBUG.md#9-比大小后的反馈手势)。此独立脚本由上层程序在比大小后调用，不自动订阅比赛结果。

反馈手势的灵巧手已默认使用 `finger_speed_mode: "max"`（目标位置＋时间 0）；机械臂为 50%。`finger_max_wait_s: 0.65` 是指令后的观察时间，不是限速参数。

## 摇晃指令下发频率

`configs/joint_shake.json` 的 `command_rate_hz=200` 表示每 5 ms 更新一次七轴 `move_js()` 目标，不是每秒摇晃 200 次。开发入口的 `calibration_debug/joint_test_config.json` 使用同一参数。支持 20–200 Hz；省略或设为 `null` 保留原来等待新反馈后发送的循环。

200 Hz 模式使用独立只读反馈线程，发送线程按单调时钟调度；反馈过期、故障及运动约束检查仍有效。迟到时跳过错过的时隙，不连续补发积压目标。Python/Linux 调度与 CAN 发送仍可能有抖动，不能把配置值当作实测频率。

摇晃 `actual.json` 中的 `command_stream` 记录 `requested_hz`、`achieved_hz`、`max_interval_ms`、`skipped_slots` 和 `max_lateness_ms`；频率基于 SDK 调用完成时间，不是 CAN 总线抓包时间。改配置后重新运行 Pipeline，独立关节测试需重新 plan。
