# 手眼标定与固定板外参恢复

`collect --preview` 人工采集法兰姿态、手背板图像及每帧七轴反馈；`solve` 求解手眼变换。`auto_collect.py draw-window` 画手背板的可见范围，`plan` 离线检查示教路径，`run --execute` 自动移动机械臂并重采图像。`reference_board.py` 注册桌面板，并在相机移动后恢复外参。具体命令和限制见 [标定文档](../docs/CALIBRATION.md)。

红布场景默认使用 USB 2.0 的 1280×720、6 FPS。仓库根目录的 `python scripts/set_camera_profile.py usb2|usb3` 同步修改手背板、固定板及抓杯检测配置；自动重采旧示教数据集时，可用 `auto_collect.py run --fps 6` 只改采样帧率。更改分辨率或裁剪后需要重新示教和标定。

## 使用入口

- [项目安装和运行](../README.md)
- [分步调试](../docs/DEBUG.md)
- [标定](../docs/CALIBRATION.md)
- [应用接入](../docs/INTEGRATION.md)
