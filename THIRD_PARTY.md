# 第三方资源

- `agx_arm_ros/src/agx_arm_description/agx_arm_urdf`：AgileX 官方机械臂与 Revo2 描述文件，随包保留其 LICENSE。该资源用于 FK、TCP 和碰撞几何，不要求启动 ROS。发布时仅规范生成的 URDF/Xacro 行尾空白，不改变几何数值。
- `pyAgxArm`：独立安装，通过 `NERO_SDK_DIR` 指定源代码目录；发行包不包含 SDK 或其虚拟环境。使用与实际 NERO 固件匹配、已经验证的版本。
- NumPy、SciPy、OpenCV、RealSense SDK、ONNX Runtime、python-can：按各自许可证安装，不将本机二进制环境复制到其他架构。
- `cup_grasp_demo/models`：项目提供的检测模型。随本次源码交付保留；未另行声明权重及训练数据许可证。

本项目未另行声明统一的开源许可证。第三方文件按各自许可证使用；公开源码不改变其许可条件。
