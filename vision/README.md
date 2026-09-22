# vision/ — 视觉子系统

五层结构（仿 yolov8_segdetect 层次感）：

| 层 | 目录 | 内容 |
|---|---|---|
| 采集 | `capture/` | `camera.json`（相机唯一配置源）、`realsense_session.py`（D435i 采集+常驻读帧+推流）、`frame_io.py`（帧数据读写）、`config.py`（配置加载器+标定绑定） |
| 推理 | `inference/` | `detector.py`（YOLO 推理+provenance）、`model_adapter.py`（模型输出自动适配）、`yolo_seg.py`（前处理+解码） |
| 几何 | `geometry/` | `circle_rim.py`（圆口立体拟合）、`cup_height.py`（深度带杯高）、`table_plane.py`（通用几何基元） |
| 策略 | `strategy/` | `green_cup.json`（抓取参数）、`loader.py`（策略加载） |

## 新物体接入指南（四步）

### ① 换模型
```bash
# 把新模型放到 cup_grasp_demo/models/<物体名>/
# 改 configs/green_cup.json 的 green_cup.perception.model 指向新路径
# 推理层自动适配：model_adapter.py 从 metadata 读取类别数和输出格式
```

### ② 写几何拟合器
```bash
# vision/geometry/ 下新增一个文件（或改配置用 circle_rim）
# 需要提供：从检测结果→杯口/接触点/桌面平面的几何映射
# 参考 circle_rim.py 的 detect_stereo() 接口
```

### ③ 写抓取策略
```bash
# vision/strategy/ 下新增 JSON（复制 green_cup.json 改参数）：
cp vision/strategy/green_cup.json vision/strategy/<物体名>.json
# 修改 contact_offset_base_mm、grip_targets_0_100 等字段
# 改 configs/green_cup.json 的 green_cup.strategy_file 指向新策略
```

### ④ 零改动阶段机
阶段机（green_pipeline.py）通过 strategy loader 读取策略，**不需要改任何流程代码**。

## camera.json 字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| `serial` | str | D435i 序列号 |
| `color_resolution` | [W,H] | 彩色流分辨率（640x480 或 1280x720） |
| `depth_resolution` | [W,H] | 深度流分辨率（640x480/848x480/1280x720） |
| `fps` | int | 帧率（6/15/30；USB 2.0 验证档为 6） |
| `crop_xywh` | [x,y,w,h] | 彩色图裁剪窗口 |
| `warmup_frames` | int | 常驻启动预热帧数（1~60） |
| `fresh_discard_frames` | int | 正式采集前丢弃帧数（0~5） |
| `calibration_file` | str | 标定结果路径（换相机配置后须重做桌面登记） |

**换配置须知**：改分辨率或裁剪后必须重新标定和桌面登记（`scripts/table_capture.py` + `scripts/register_home_table.py`），camera.json 的 `calibration_file` 绑定会校验标定一致性。

## 使用 set_camera_profile.py 切换相机档位
```bash
python3 scripts/set_camera_profile.py usb2   # 1280×720@6（默认，USB 2.0 验证档）
python3 scripts/set_camera_profile.py usb3   # 1280×720@15
python3 scripts/set_camera_profile.py usb2 --fps 6 --color-resolution 640 480
```
