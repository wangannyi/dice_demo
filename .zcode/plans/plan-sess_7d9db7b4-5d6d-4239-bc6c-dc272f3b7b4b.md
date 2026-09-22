# 视觉分层重构计划（最终版）：先清孤儿 → vision/ 五层 + camera.json + 策略模块化

## Phase 0：清理视觉孤儿（1 提交）

删（全部经引用扫描确认为零代码引用）：
- `cup_grasp_demo/config/` 整目录（5 个上游遗留 JSON：green_pipeline.json、green_top_retracted_thumb.json、3×pipeline_restart_*.json）
- `side_grasp/`：prepare_hand.py + 其测试、STATUS.json、TCP_MODEL_CANDIDATE.json、superseded_candidates.json
- `dice_cup_localization/`：config/（red_mat_640x480.json，只被上面孤儿引用）、README.md、requirements.txt

保留（import 闭包 + 引用扫描确认存活）：dice_cup_localization 的 3 个 py、side_grasp/preview_index.py + CURRENT_GRASP.json（green_cup.json 的 grasp_config 指向它）、cup_grasp_demo 顶层的 hand_geometry.py/planning.py/models//datasets/

## Phase A：建 vision/ 骨架 + 文件迁移（1 提交）

存档先行（git commit --allow-empty），然后：
- `dice_cup_localization/capture_rgbd.py` → `vision/capture/realsense_session.py`
- `side_grasp/preview_index.py` 的 load_batch/section → `vision/capture/frame_io.py`
- `green_yolo.py` 的 session/runtime_settings → `vision/inference/session.py`
- `green_yolo.py` 的 cap_outputs/infer 逻辑 → `vision/inference/detector.py`
- `dice_cup_localization/yolo_seg.py` → `vision/inference/yolo_seg.py`
- `green_stereo_rim.py` → `vision/geometry/circle_rim.py`
- `green_cup_geometry.py` → `vision/geometry/cup_height.py`
- `dice_cup_localization/geometry.py` 的 _plane/deproject/_circle/Config → `vision/geometry/table_plane.py`
- `dice_cup_localization/` 目录清空后删除
- 全局 import 路径替换（约 15 个消费方）

## Phase B：相机配置独立 vision/camera.json（1 提交）

```json
{
  "serial": "346222071954",
  "color_resolution": [1280, 720],
  "depth_resolution": [1280, 720],
  "fps": 6,
  "crop_xywh": [220, 0, 960, 720],
  "warmup_frames": 5,
  "fresh_discard_frames": 0
}
```
- `green_capture.py`/`green_runtime.py` 改读 camera.json（不再读 green_cup.json 的 camera 段）
- green_cup.json 删 green_cup.camera 段 + fast_camera_warmup_frames/fast_camera_fresh_discard_frames
- `set_camera_profile.py` 改写 camera.json（不再同步三份配置）
- camera.json 加 calibration_file 字段指向 configs/calibration/handeye_result.json，加载时校验 sha256（换相机配置须重新标定的安全绑定）

## Phase C：策略配置模块化 + 模型适配器（1 提交）

- `vision/strategy/green_cup.json`：从 green_cup.json 抽出全部抓取字段（contact_offset_base_mm/tcp_offset_flange_mm/wrist_reference_deg/lift_mm/grip/open_targets/finger_duration_s/held_cup_margin_mm 等）
- `vision/strategy/loader.py`：返回 GraspStrategy 命名元组
- green_pipeline.py 改用 strategy.loader
- `vision/inference/model_adapter.py`：从 ORT metadata 解析 names 字段自动适配类别数/输出 shape（现有 cap/ground 双类模型的行为不变，新模型自动识别）

## Phase D：测试迁移 + 文档 + 全量验证（1 提交）

- 测试文件迁移到 tests/vision/ 镜像结构
- `vision/README.md` 四步接入指南（换模型→写几何拟合器→写策略 JSON→零改阶段机）+ camera.json 字段说明
- 全量验证四连：pytest 集合对比零新增 + check_environment + fast 干跑 + simulate 冒烟

## 验证关卡（每 Phase）

- 每步定向 pytest + py_compile
- import 闭包六入口全通
- 最终全量与基线（8F/29E）集合对比零新增

## 提交：Phase 0/A/B/C/D 各 1 提交，每 Phase 前存档（hwj_dev）