# K3 运行环境与依赖清单

核对日期：2026-09-22（主 K3 部署实测，路径 `~/projects/dice-game/dice_demo`）。本板不使用虚拟环境：`scripts/env.sh` 探测 `$HOME/.venv-grasp`/`$HOME/agilex-api-test` 均不存在时自动回退系统 `/usr/bin/python3` + 仓库自带依赖（`vendor-site/`、`nero_calibration/.deps/`）。版本代表当前安装状态，不代表所有板卡必须使用这些版本。安装入口见[顶层 README](../README.md#2-安装运行环境)。

## 1. 系统与硬件运行条件

| 项目 | 当前环境 | 用途 |
| --- | --- | --- |
| 操作系统 | Bianbu 4.0，`resolute` | Debian/Ubuntu 系包管理环境 |
| 架构 | `riscv64` | 二进制包必须匹配 RISC-V，不能使用 PC 的 x86_64 包 |
| 内核 | `6.18.3-gsusb` | 当前 USB CAN 驱动环境 |
| CPU | 16 个逻辑核，SpacemiT A100/X100 | 当前 YOLO 配置指定 CPU 8、9，2 个推理线程 |
| 系统解释器 | `/usr/bin/python3.14`，Python 3.14.4 | 两个虚拟环境的基础解释器 |
| CAN | Linux SocketCAN；`gs_usb`；`can0`，1,000,000 bit/s | NERO 和 Revo2 通信 |
| 相机 | Intel RealSense D435i；已在 `480M` USB 2.0 下完成 1280×720、6 FPS 采集和只读定位测试 | 当前抓杯与红布标定板默认使用该档位；切到 USB 3.0 可配置 1280×720、15 FPS |
| 预览 | X11，经 SSH 转发到 PC | STEP/标定窗口；FAST 不要求显示窗口 |

`python3` 的 Debian 元包版本为 `3.14.3-0ubuntu2`，实际解释器报告 `3.14.4`；判断 Python 扩展 ABI 时以解释器版本为准。

当前内核 `6.18.3-generic-usbcan`（USB-CAN 驱动已内建，`gs_usb` 等模块齐全）。历史构建工具目录已随 2026-09-22 瘦身移除。

### 系统库和工具

| 依赖 | 当前包/版本 | 适用范围 |
| --- | --- | --- |
| USB 运行库 | `libusb-1.0-0`：`2:1.0.29-2build1` | USB 设备支持；RealSense wheel 还携带自己的动态库 |
| C/C++ 运行库 | `libc6`、`libstdc++6`、`libgcc-s1` | Python 原生扩展、RealSense、ORT |
| OpenMP | `libgomp1`：`16-20260226-1ubuntu1` | 数值库运行依赖 |
| BLAS | `libopenblas0-pthread`：`0.3.32+ds-5`；`openblas-spacemit`：`0.3.32-1bb2` | 板端数值计算库 |
| OpenCV 图形依赖 | `libgl1`：`1.7.0-3`；`libglib2.0-0t64`：`2.87.3-1` | OpenCV 导入及显示 |
| X11/Qt 支持 | `libxkbcommon-x11-0`：`1.13.1-1`；`libxcb-xinerama0`：`1.17.0-2ubuntu1`；`xauth`：`1:1.1.2-1.1build1` | 预览窗口 |
| SSH | `openssh-server`：`1:10.2p1-2ubuntu1` | PC 登录、X11 转发 |
| 网络接口管理 | `iproute2`：`6.18.0-1ubuntu1` | 查询/配置 CAN 接口 |
| USB 查询 | `usbutils`：`1:019-1` | `lsusb`、USB 链路速度检查 |
| 相机调试 | `v4l-utils`：`1.32.0-2ubuntu1bb1`；`ffmpeg`：`7:8.0.1-3ubuntu2bb1` | 可选的 v4l2/ffplay 调试工具 |
| 源码/构建 | Git、`build-essential`、CMake、`pkg-config` | 拉取源码及构建本机扩展；已构建运行包不需要每次编译 |

`can-utils` 不在本次已安装包清单中，`candump` 也未找到。它是额外的 CAN 诊断工具，不是当前 pipeline 的运行前提。

## 2. Python 环境分工

统一使用系统 Python **3.14.4**（`/usr/bin/python3`）；视觉与机械臂执行共用同一解释器，差异只在 `PYTHONPATH` 注入的仓库自带依赖。

| 环境 | 实际解释器 | 执行内容 |
| --- | --- | --- |
| 视觉/标定/执行 | `/usr/bin/python3`（env.sh 回退链终点） | pipeline 状态机、相机、YOLO、IK、标定、持久 SDK 子进程、SocketCAN、`move_js`、手指指令 |

`run.sh` 会加载 `scripts/env.sh` 并显式选择解释器（探测顺序：`$HOME/.venv-grasp` → `$HOME/agilex-api-test/venv` → `/usr/bin/python3`），不要求先 `source activate`。

### 依赖分布（2026-09-23 实测）

| 层 | 包 | 来源 |
| --- | --- | --- |
| 系统 apt（/usr/lib/python3/dist-packages） | numpy 2.3.5、scipy 1.16.3、cv2 4.10（含 aruco） | `python3-numpy` / `python3-scipy` / `python3-opencv` |
| 系统 apt（/usr/lib/python3.14/dist-packages） | onnxruntime 1.24.2+spacemit.a1、spacemit_ort 2.0.6 | `spacemit-onnxruntime` / `python3-spacemit-ort`（K3 厂商源） |
| 仓库 vendor-site/（27MB，gitignored） | pyrealsense2 2.57.7、pyAgxArm（NERO SDK 源码）、packaging、wrapt | 已集成到当前目录，无需外部安装 |
| 仓库 nero_calibration/.deps/（3MB，gitignored） | python-can 4.6.1、typing_extensions | 已集成到当前目录 |

### 视觉环境的主要 Python 依赖

| 包 | 当前版本 | 实际来源/用途 |
| --- | --- | --- |
| NumPy | `2.3.5` | apt `python3-numpy`（`/usr/lib/python3/dist-packages`）；矩阵计算 |
| SciPy | `1.16.3` | apt `python3-scipy`；IK、优化和几何 |
| OpenCV | 模块 `4.10.0` | apt `python3-opencv`（绑定层含 `cv2.aruco`）；图像处理 |
| pyrealsense2 | `2.57.7` | 仓库 `vendor-site/`；D435i 采集 |
| ONNX Runtime | `1.24.2+spacemit.a1` | `/usr/lib/python3.14/dist-packages/onnxruntime`；厂商构建 |
| spacemit-ort | `2.0.6` | `/usr/lib/python3.14/dist-packages/spacemit_ort`；注册 SpaceMIT 推理后端 |
| python-can | `4.6.1` | 通过 `PYTHONPATH` 优先使用 `nero_calibration/.deps/can`；标定反馈读取 |
| typing_extensions | `4.16.0` | `nero_calibration/.deps`；SDK 兼容依赖 |
| pyAgxArm | 包版本 `1.0.0` | 仓库 `vendor-site/pyAgxArm`；标定和 SDK API |

`pyrealsense2` 不提供本次可读的模块 `__version__`，上表版本来自发行包元数据。其原生扩展为 `cpython-314-riscv64-linux-gnu.so`；`pyrealsense2.libs` 内携带 librealsense2、libusb、libudev，不能只复制一个 Python 文件或一个 `.so`。

### CAN 执行环境的主要 Python 依赖

| 包 | 当前版本 | 说明 |
| --- | --- | --- |
| python-can | `4.6.1` | SDK 环境内有安装；加载 `scripts/env.sh` 后 `.deps` 中同版本优先 |
| typing_extensions | `4.16.0` | SDK 环境内有安装；同样可被 `.deps` 覆盖 |
| pyAgxArm | `1.0.0` | `NERO_SDK_DIR` 的源码优先于虚拟环境内安装副本 |
| packaging / wrapt | `26.3` / `1.17.3` | 已安装的辅助依赖 |

当前 SDK 环境没有安装 NumPy、SciPy、OpenCV 或 pyrealsense2，标准 CAN 执行路径不靠它们运行。`requirements-sdk.txt` 仍列有 NumPy，它是现有安装清单中的额外项，不能据此推断板端 SDK 环境已安装。此次仅记录现状，没有修改依赖清单或安装包。

## 3. 虚拟环境之外的依赖

### pyAgxArm 源码

- 目录：仓库内 `vendor-site/pyAgxArm`（env.sh 的 `NERO_SDK_DIR` 默认指向此处；`$HOME/agilex-api-test/pyAgxArm` 存在时优先）。
- 实测提交：`e7aef17d54cac80cbaeb1b4110ab3d8f1337a95b`。
- 本次 `git status --short` 无输出。
- 两个环境都通过 `NERO_SDK_DIR` 和 `PYTHONPATH` 使用该源码。

包版本 `1.0.0` 不能唯一标识源码内容；交接时还需保留该提交及与 NERO 固件匹配的 SDK。

### SpacemiT ONNX Runtime

已安装的系统包为 `spacemit-onnxruntime=2.0.6`、`python3-spacemit-ort=2.0.6`。Python 模块的 ORT 版本号是 `1.24.2+spacemit.a1`，与系统包版本号不同。

`configs/green_cup.json` 当前配置：

```json
{
  "ort_package_dir": "/usr/lib/python3.14/dist-packages",
  "inference_provider": "spacemit",
  "inference_threads": 2,
  "inference_cpu_ids": [8, 9]
}
```

这些字段属于 `green_cup.perception`，不是完整配置文件。推理代码先导入 `spacemit_ort` 注册后端；注册后实测提供 `SpaceMITExecutionProvider` 和 `CPUExecutionProvider`。只导入 `onnxruntime` 时可能只看到 CPU 后端。

### 项目补充依赖目录

`nero_calibration/.deps` 经 `PYTHONPATH` 注入，位于虚拟环境 `site-packages` 之前。当前有效的 `can` 和 `typing_extensions` 就来自这里。安装了同名包后，仍应检查模块 `__file__`，确认程序实际加载哪个副本。

### 不是当前抓杯 pipeline 的必需项

ROS 2 / MoveIt、MediaPipe、TensorFlow、PyTorch、Ultralytics Python 运行库和语音 ASR/TTS/VAD 不属于当前绿杯 ONNX pipeline 的必需运行栈。板端安装了部分 ROS/语音软件，不应把整机包列表全部当作本项目依赖。

## 4. 启动脚本选择与环境变量

| 变量 | 默认值/行为 |
| --- | --- |
| `DICE_VISION_PYTHON` | 探测回退链终点 `/usr/bin/python3` |
| `DICE_SDK_PYTHON` | 同上 |
| `NERO_SDK_DIR` | `vendor-site/pyAgxArm` |
| `CALIB_PYTHON` | 默认跟随视觉解释器 |
| `PYTHONPATH` | 加入项目根目录、SDK 源码、`nero_calibration/.deps` |
| `OPENBLAS_NUM_THREADS` | `1` |
| `PYTHONNOUSERSITE` | `1`，不加载用户级 site-packages |
| `QT_X11_NO_MITSHM` | `1`，用于 X11 预览兼容 |

`nero_calibration/run_k3.sh` 单独使用 `CALIB_PYTHON`，默认同样走 env.sh 回退链（本板为系统 python）。

## 5. 环境核验命令

以下命令只读取环境，不发送机械臂动作：

```bash
cd ~/projects/dice-game/dice_demo
source scripts/env.sh
bash scripts/check_environment.sh
"$DICE_VISION_PYTHON" --version
"$DICE_SDK_PYTHON" --version
uname -r
uname -m
lsusb -t
ip -details link show can0
```

相机参数由仓库根目录的 `python scripts/set_camera_profile.py usb2|usb3` 统一切换；`--fps` 与 `--color-resolution`、`--depth-resolution` 可指定受支持的档位。`lsusb` 必须能列出 D435i，才能进行实拍验证；只通过环境检查或离线测试不能证明相机当前在线。完整命令及重新标定条件见[顶层 README](../README.md#相机配置)。

`check_environment.sh` 验证核心包导入、ArUco、配置路径、检测模型和几何模型；不打开相机或 CAN。本次已通过。该脚本没有创建 SpaceMIT 推理会话，不能单独证明加速后端推理成功。

模块来源可通过各解释器的 `模块.__file__` 核对。迁移时需同时保留系统厂商包、RISC-V/Python 3.14 扩展构建产物、SDK 源码、`.deps`、模型与配置；`vendor-site/` 与 `nero_calibration/.deps` 的包清单只是其中一部分。仓库 `requirements-vision.txt` 和 `requirements-sdk.txt` 尚不是完整可复现的版本锁文件。
