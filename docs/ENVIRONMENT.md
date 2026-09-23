# K3 运行环境

## 1. 硬件与系统

| 项目 | 要求 |
| --- | --- |
| 开发板 | SpacemiT K3，RISC-V 64 位 |
| 机械臂 | AgileX NERO 七轴 |
| 灵巧手 | 右 Revo2 |
| 相机 | Intel RealSense D435i |
| CAN | `can0`，1 Mbps |
| Python | K3 系统 Python 或开发板上的兼容虚拟环境 |

相机支持 USB 2.0 和 USB 3.0：

- USB 2.0 默认使用 1280×720、6 FPS。
- USB 3.0 默认使用 1280×720、15 FPS。

用以下命令确认设备和实际链路：

```bash
uname -a
python3 --version
lsusb
lsusb -t
ip -details link show can0
```

`lsusb -t` 中 `480M` 表示 USB 2.0，`5000M` 表示 USB 3.0。

## 2. 系统包

```bash
sudo apt update
sudo apt install python3-numpy python3-scipy python3-opencv \
  spacemit-onnxruntime python3-spacemit-ort
```

可选工具：

```bash
sudo apt install xauth can-utils
```

`xauth` 用于 SSH X11 预览；`can-utils` 用于 `candump` 等总线诊断，不是 Pipeline 的运行依赖。

## 3. Python 依赖

项目按功能使用以下依赖：

| 功能 | 主要依赖 |
| --- | --- |
| 数值与几何 | NumPy、SciPy |
| 标定与图像处理 | OpenCV，需包含 ArUco |
| RGB-D 相机 | `pyrealsense2` |
| 绿杯分割 | ONNX Runtime、SpacemiT ORT |
| CAN | `python-can` |
| NERO/Revo2 | `pyAgxArm` 源码 |

K3 是 RISC-V 架构。不要复制 PC 的 x86 虚拟环境，也不要假设 PyPI 提供所有 riscv64 轮子。交付包可携带：

```text
vendor-site/          pyrealsense2、pyAgxArm 等板端依赖
vendor-site-deps/     python-can、typing_extensions 等补充包
calibration/.deps/    标定工具补充依赖
```

`requirements-vision.txt` 和 `requirements-sdk.txt` 用于说明上游 Python 依赖，不是 K3 的完整安装命令。

## 4. 环境变量

在仓库根目录执行：

```bash
source scripts/env.sh
```

脚本设置：

| 变量 | 用途 |
| --- | --- |
| `DICE_ROOT` | 仓库绝对路径 |
| `DICE_VISION_PYTHON` | 相机、推理、几何和规划解释器 |
| `DICE_SDK_PYTHON` | CAN 和灵巧手执行器解释器 |
| `NERO_SDK_DIR` | `pyAgxArm` 源码目录 |
| `CALIB_PYTHON` | 标定解释器 |
| `PYTHONPATH` | 仓库、SDK 和随包依赖的加载路径 |

默认探测顺序：

1. 视觉：`$HOME/.venv-grasp/bin/python`，否则 `/usr/bin/python3`。
2. SDK：`$HOME/agilex-api-test/venv/bin/python`，否则 `/usr/bin/python3`。
3. NERO SDK：`$HOME/agilex-api-test/pyAgxArm`，否则 `vendor-site/pyAgxArm`。

需要指定其他环境时，在 `source` 前设置变量：

```bash
export DICE_VISION_PYTHON=/path/to/python
export DICE_SDK_PYTHON=/path/to/python
export NERO_SDK_DIR=/path/to/pyAgxArm
source scripts/env.sh
```

## 5. CAN 和 WEB 设置

启动 CAN：

```bash
sudo ip link set can0 up type can bitrate 1000000
ip -details link show can0
```

同时确认：

- 急停已解除。
- 七个机械臂关节已使能。
- WEB 控制页已选择 Revo2 并打开灵巧手使能。
- 控制器已开启 CAN 推送。
- 没有其他程序持有机械臂 SDK 或 CAN 接收器。

Linux 接口显示 `UP` 只表示 SocketCAN 已启动，不代表控制器已经处于 CAN 模式。

## 6. 相机配置

```bash
python3 scripts/set_camera_profile.py usb2
python3 scripts/set_camera_profile.py usb3
python3 scripts/set_camera_profile.py usb2 --dry-run
```

配置工具同步修改抓杯和标定的相机参数。更改彩色分辨率或裁剪后需重新标定；只改变帧率时可保留空间标定，但必须重新验证相机采集和杯位。

X11 预览：

```bash
ssh -X user@k3-host
export QT_X11_NO_MITSHM=1
echo "$DISPLAY"
```

## 7. 环境验证

```bash
source scripts/env.sh
bash scripts/check_environment.sh
```

该脚本检查 Python 导入、ArUco、配置路径、模型和几何文件，不打开相机或 CAN。进一步检查：

```bash
# 不访问硬件
bash run.sh fast
python3 scripts/control_console.py --simulate

# 相机检测，不移动机械臂
source scripts/env.sh
bash cup_grasp_demo/flow/run_debug.sh \
  green-detect \
  --config configs/green_cup.json \
  --session cup_grasp_demo/datasets/green_current
```

新安装验证顺序：环境检查 → 标定 → 桌面登记 → 杯子检测 → Pipeline 预览 → 真机运行。
