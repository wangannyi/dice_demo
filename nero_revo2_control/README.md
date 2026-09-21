# NERO 与右 Revo2 单项控制

本模块提供 SocketCAN 控制、七轴反馈、FK、关节运动和六路灵巧手指令。当前抓杯摇骰使用[顶层入口](../README.md)，上层应用调用方式见[接口文档](../docs/INTEGRATION.md)。

## 1. 环境

使用顶层环境安装和 `scripts/check_environment.sh`。需与固件匹配的 pyAgxArm、python-can，以及已启用的 1 Mbps can0。机械臂七轴使能与 WEB 灵巧手使能是不同设置。运行期间只保留一个控制程序。

## 2. 交互控制

从仓库根目录执行：

```bash
source scripts/env.sh
cd nero_revo2_control
"$DICE_SDK_PYTHON" nero_revo2_demo.py
```

菜单提供机械臂状态、七轴角度、关节/笛卡尔运动、FK、手状态、手部控制和 ready-home。填写动作参数后直接执行；需要逐次确认时添加 `--confirm`。单项 demo 的 ready-home 与 Pipeline 配置中的 HOME 不应视为同一个姿态。

## 3. 命令行与只读检查

```bash
"$DICE_SDK_PYTHON" nero_revo2_demo.py --help
"$DICE_SDK_PYTHON" nero_revo2_demo.py --format json hand-status
```

各子命令用 `--help` 查看参数。命令行运动是否执行以该子命令的 `--execute` 参数为准。关节角输入明确区分度与弧度；手指六路顺序为拇指尖、拇指根、食指、中指、无名指、小指。

## 4. 运动学模型

`models/nero_description.urdf` 用于机械臂运动学，终点 link7 为模型法兰参考；保留原模型 [MIT 许可证](models/LICENSE)。手部 TCP 需叠加安装变换及当前配置的偏移。FK 和 IK 数值通过不等于真实场景碰撞或接触精度验收。

标定及 TCP 配置见[标定指南](../docs/CALIBRATION.md)，逐阶段检查见[调试指南](../docs/DEBUG.md)。
