# K3 运行环境

## 1. 平台

| 项目 | 要求 |
| --- | --- |
| 开发板 | SpacemiT K3，RISC-V 64 位 |
| 机械臂 | AgileX NERO 七轴 + 右 Revo2 |
| 相机 | Intel RealSense D435i |
| CAN | `can0`，1 Mbps |
| Python | 系统 `python3`，当前 K3 使用 Python 3.14 |

D435i 支持 USB 2.0 和 USB 3.0。项目的 USB 2.0 档位为 1280×720、6 FPS；USB 3.0 档位为 1280×720、15 FPS。

```bash
uname -a
python3 --version
lsusb -t
ip -details link show can0
```

`lsusb -t` 中 `480M` 表示 USB 2.0，`5000M` 表示 USB 3.0。

## 2. 新 K3 一键安装

克隆仓库后执行：

```bash
sudo bash scripts/bootstrap_k3.sh
source scripts/env.sh --system
bash scripts/check_environment.sh
```

安装脚本会先检查 `riscv64`、CPython 3.14 和随仓库 wheel 的 SHA256，然后通过 Bianbu 安装通用系统包，并从系统 Python 的模块搜索路径中自动选择 `/usr/local` 下的安装目录来安装 RealSense 扩展，最后运行无硬件环境检查。它不会创建虚拟环境，也不会打开相机或 CAN。

### 识别加速依赖锁定

识别使用 `configs/k3_runtime.lock.json` 中锁定的依赖：

| 包 | 固定版本 |
| --- | --- |
| `spacemit-onnxruntime` | `2.0.6` |
| `python3-spacemit-ort` | `2.0.6` |
| `spacemit-tcm` | `3.0.0+5`（库文件版本 `3.0.0`） |

安装脚本将这些包解压至本仓库的 `runtime/`，校验包和库文件的 SHA256，不替换系统共享库。APT 中的系统版本可以保持较新，语音服务继续使用其现有环境。仓库不提交解压后的二进制。

已安装通用依赖的 K3 可只安装或核对锁定库：

```bash
python3 scripts/k3_runtime.py install
source scripts/env.sh --system
bash scripts/check_environment.sh
```

安装优先使用 `/var/cache/apt/archives`，缺包时只下载锁文件中的精确版本；版本不可用或校验失败会停止，不会自动改用最新版。`run.sh`、`calibrate.sh` 通过 `env.sh` 自动加载锁定库，普通重启仍生效。库缺失或发生变化时，启动会报错并给出安装命令。

上层应用启动原生 YOLO 裁决进程时，可执行 `python3 scripts/k3_runtime.py environment --native` 获取 JSON 格式的 `LD_LIBRARY_PATH` 和 `LD_PRELOAD`，仅合入该视觉子进程的环境。Python 杯子识别与原生 YOLO 分别使用各自软件包中的 ORT 核心，避免混用。不要把这些环境变量设置到整个上层应用或 TTS 服务。

需要升级时修改锁文件中的精确版本及校验值，完成连续游戏回归后再发布；`apt upgrade` 不会改变本仓库已安装的识别运行库。

## 3. 系统 Python 依赖

项目不要求创建虚拟环境，默认使用 `PATH` 中的 `python3`。在 K3 系统镜像中安装以下包：

- NumPy、SciPy
- OpenCV，必须包含 ArUco
- pyrealsense2，版本必须匹配系统架构和 Python ABI
- ONNX Runtime；K3 使用 SpacemiT ORT
- python-can、wrapt、packaging、typing-extensions

K3 系统仓库可用时，优先通过系统包管理器安装：

```bash
sudo apt update
sudo apt install python3-numpy python3-scipy python3-opencv \
  python3-can python3-wrapt python3-packaging python3-typing-extensions
python3 scripts/k3_runtime.py install
```

`pyrealsense2` 包含与架构和 CPython 版本绑定的二进制扩展。仓库附带的 wheel 只适用于 K3 的 riscv64/CPython 3.14，安装脚本会拒绝不匹配的平台。

NERO/Revo2 的 `pyAgxArm` 已固定在仓库的 `third_party/pyAgxArm/`，不需要外部 SDK 目录。

## 4. 环境初始化

可以从任意目录加载：

```bash
source /path/to/dice_demo/scripts/env.sh --system
```

脚本根据自身位置计算仓库根目录，不包含安装机器的绝对路径。`--system` 会退出当前虚拟环境、清除遗留的 `DICE_*`/`PYTHONPATH` 覆盖，并明确选择 `/usr/bin/python3`。默认设置如下：

| 变量 | 默认值 |
| --- | --- |
| `DICE_ROOT` | 当前仓库根目录 |
| `DICE_PYTHON` | `PATH` 中的 `python3` |
| `DICE_VISION_PYTHON` | `DICE_PYTHON` |
| `DICE_SDK_PYTHON` | `DICE_PYTHON` |
| `CALIB_PYTHON` | `DICE_VISION_PYTHON` |
| `NERO_SDK_DIR` | `third_party/pyAgxArm` |

需要显式覆盖时，在加载脚本之前设置变量：

```bash
export DICE_PYTHON=/path/to/python3
# 可选：额外的项目本地 site-packages 目录
export DICE_PYTHON_EXTRA=/path/to/site-packages
source scripts/env.sh
```

`DICE_PYTHON_EXTRA` 是部署接口，不是固定安装路径。普通安装不应设置它。

K3 的动态库路径由 `scripts/runtime_library_path.py` 根据 `DICE_VISION_PYTHON`
中的 `onnxruntime` 和 `spacemit_ort` 自动生成。两者必须来自同一个 Python
包目录。脚本在系统临时目录创建用户专属的库别名，使插件依赖的
`libonnxruntime.so`、`libonnxruntime.so.1` 指向 Python 绑定配套的同一个核心库，
随后加入该包的 `capi`、插件目录及 `/usr/lib`。已有 `LD_LIBRARY_PATH` 放在后面并去重，
避免 `/usr/local/lib` 的旧推理库抢先被加载；其他架构保持原有加载路径。
这一步不安装或升级系统包。库更新后，下次启动会自动选择新的别名目录。

## 5. 环境验证

```bash
source scripts/env.sh --system
bash scripts/check_environment.sh
```

检查脚本会输出每个模块的版本和实际来源，验证 ArUco、配置、模型、几何文件和仓库内的 `pyAgxArm`，不会打开相机或 CAN。所有模块都应由系统目录或当前仓库提供，不应来自旧项目目录或用户虚拟环境。
使用 SpaceMIT 推理时，还会核对进程实际加载的 ONNX Runtime 核心库，混入其他版本会使检查失败。

进一步执行无硬件预览：

```bash
bash run.sh fast
python3 scripts/control_console.py --simulate
```

## 6. CAN 和 WEB

```bash
sudo ip link set can0 up type can bitrate 1000000
ip -details link show can0
```

同时确认急停已解除、七轴已使能、WEB 页面已选择 Revo2 并打开灵巧手使能和 CAN 推送。Linux 接口显示 `UP` 只表示 SocketCAN 已启动，不代表控制器已处于 CAN 模式。

## 7. 相机档位

抓杯相机参数在 `vision/camera.json`；标定采样帧率在
`configs/calibration_workflow.json` 的 `fps`，板文件的 `image_profile`
设置分辨率与裁剪。保持抓杯与标定使用的相机档位一致。
改变彩色分辨率或裁剪后必须重新标定。只改变帧率时可以保留空间标定，但仍需重新验证采集和杯位。

新安装的验证顺序：环境检查 → 标定 → 桌面登记 → 杯子检测 → Pipeline 预览 → 真机运行。
