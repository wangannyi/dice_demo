# 第三方资源

- `nero_revo2_control/models/hand_geometry`：AgileX 官方机械臂与 Revo2 描述文件，随包保留其 LICENSE。该资源用于 FK、TCP 和碰撞几何，不要求启动 ROS。发布时仅规范生成的 URDF/Xacro 行尾空白，不改变几何数值。
- `third_party/pyAgxArm`：AgileX 官方 Python SDK，固定到提交 `e7aef17d54cac80cbaeb1b4110ab3d8f1337a95b`，按 LGPL-3.0-only 分发并保留上游 LICENSE。项目通过仓库相对路径加载它。
- NumPy、SciPy、OpenCV、ONNX Runtime、python-can、wrapt、packaging、typing-extensions：使用目标系统的软件包。
- `third_party/wheels/k3-cp314/pyrealsense2-2.57.7-*.whl`：K3 riscv64/CPython 3.14 专用安装产物，SHA256 和平台标签记录在同目录 `MANIFEST.json`，按 Apache-2.0 使用并保留官方 LICENSE。它经过隔离安装、导入及动态库检查；安装脚本会拒绝其他平台。
- `cup_grasp_demo/models`：项目提供的检测模型。随本次源码交付保留；未另行声明权重及训练数据许可证。

本项目未另行声明统一的开源许可证。第三方文件按各自许可证使用；公开源码不改变其许可条件。
