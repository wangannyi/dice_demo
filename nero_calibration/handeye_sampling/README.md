# 手眼标定采样

人工采样入口为 `nero_calibration/run_k3.sh collect --preview`。预览的每一帧将七轴反馈写入 `teaching_frames.jsonl`；有效样本的采集帧反馈写入各 `sample_*.json`。人工改变姿态后静止采样，保存的数据、`teaching_poses.json` 和所画的 `board_window.json` 应一起备份。重复采样使用 `nero_calibration/auto_collect.py plan` 离线检查，再用 `run --execute` 自动移动并采图；见 [标定文档](../../docs/CALIBRATION.md)。

## 使用入口

- [项目安装和运行](../../README.md)
- [分步调试](../../docs/DEBUG.md)
- [标定](../../docs/CALIBRATION.md)
- [应用接入](../../docs/INTEGRATION.md)
