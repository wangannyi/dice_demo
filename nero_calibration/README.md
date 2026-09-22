# 手眼标定与固定板外参恢复

`collect --preview` 人工采集法兰姿态、手背板图像及每帧七轴反馈；`solve` 求解手眼变换。`auto_collect.py draw-window` 画手背板的可见范围，`plan` 离线检查示教路径，`run --execute` 自动移动机械臂并重采图像。`reference_board.py` 注册桌面板，并在相机移动后恢复外参。具体命令和限制见 [标定文档](../docs/CALIBRATION.md)。

## 使用入口

- [项目安装和运行](../README.md)
- [分步调试](../docs/DEBUG.md)
- [标定](../docs/CALIBRATION.md)
- [应用接入](../docs/INTEGRATION.md)
