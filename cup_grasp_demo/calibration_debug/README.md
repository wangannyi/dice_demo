# Pipeline 与调试命令

run_debug.sh 提供 pipeline、green-detect、config-check、tcp-view 及历史对点命令；run_joint_test.sh 提供独立关节往复；run_planar_shake.sh 提供平面摇晃实验。新用户从顶层入口开始，勿将历史银杯配置直接用于绿杯流程。

绿杯检测默认使用 USB 2.0 的 1280×720、6 FPS。切换 USB 2.0/3.0 及设置帧率、分辨率时，从仓库根目录运行 `python scripts/set_camera_profile.py usb2|usb3`；命令会一并更新标定板配置。具体参数和重新标定条件见[项目安装和运行](../../README.md#相机配置)。

## 使用入口

- [项目安装和运行](../../README.md)
- [分步调试](../../docs/DEBUG.md)
- [标定](../../docs/CALIBRATION.md)
- [应用接入](../../docs/INTEGRATION.md)
