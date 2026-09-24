# 待办优化清单

> 2026-09-24 整体审计产出。做一项划掉一项；完成后在本文件底部记录提交号。

## 🎯 值得做（优先）

### 1. 测试配置与主配置漂移收敛（最大技术债）

**现状**：`cup_grasp_demo/flow/green_open_cup/config.json`（green_cup 32 键）与
`stereo_config.json`（45 键）**只被测试引用**（5 个测试文件），与交付配置
`configs/green_cup.json` 已漂移 **21 个键**：

- 测试配置缺 `fast_motion_profile: trapezoid`（跑的是缺省 quintic）
- 测试配置缺 `joint_delivery.feedback_freshness_limit_s`（2026-09-24 新增）
- `finger_duration_s`：测试 1.0 vs 主配置走 strategy 注入 0.5
- `open_targets_0_100` 等键已随策略拆分移入 `vision/strategy/green_cup.json`，
  测试配置仍是旧布局

**风险**：主配置改了行为，测试不会跟着发现——测试在验证一套越来越偏离交付的
参数组合。

**方案**：测试改为「加载 `configs/green_cup.json` + 测试专属覆盖项」派生，
删除两份静态副本。

**涉及**（5 个测试文件）：
- tests/cup_grasp_demo/flow/test_green_direct_approach.py
- tests/cup_grasp_demo/flow/test_green_pipeline.py
- tests/cup_grasp_demo/flow/test_green_plane_config.py（test_plane_config.py）
- tests/cup_grasp_demo/flow/test_green_fast_overhead.py
- tests/cup_grasp_demo/flow/test_green_image_rim.py

**注意**：收敛前先摸清每个测试依赖旧值的断言（如 quintic 缺省、finger 1.0），
逐个确认是"测试专属设定"还是"历史漂移"——前者保留为显式覆盖项，后者删除。
`green_open_cup/config.json` 同时是 green-detect 的 CLI 默认配置（debug.py），
删除需同步改默认指向主配置。`package_release.py` 会改写 stereo_config.json
的 calibration 路径，收敛时一并处理。

### 2. README 补新功能说明（顺手项）

连跑（`g5`/`g10`，rounds 协议）与 SDK worker 零传输自愈两个用户可见功能，
INTEGRATION 协议文档已有，README 一句没提。在 §3 运行模式表和 §5 附近各补
一段。

## ⏸️ 待条件成熟

- **UX 三小尾件**：手势执行期间心跳反馈（action_running 事件）；reload 后
  控制台菜单/快捷键联动重建（现需重启控制台才有新键位）；`status` 命令即时
  返回（现排队在执行队列里）。价值中低，连跑的 round_started 已部分缓解静默问题。
- **model_adapter.py 接线**：`vision/inference/model_adapter.py` 已入库但
  `detector.py` 仍硬编码 [1,38,8400] cap/ground 契约。接新物体时一并做。
- **SHAKE 起点静止"重读窗口"**：joint_execution.py 的 0.05° 起点检查失败前
  有限重读（~100ms×10）。等真机验证同事 recovery_attempts=2 的恢复逻辑
  （07678e2）后决定是否还需要。

## 🔒 需拍板（涉及协作/历史/数据）

- **.git 76.6MB 历史包袱**：大头是已删除的 Piper 四型臂网格与旧模型。不改写
  历史减不掉；真要瘦需新建 squash 干净仓库重启，须与同事共同拍板。
- **../biaoding 与 calibration/ 双源**（各约 4MB）：同事拷贝交付造成，两份并存
  会漂移。等同事确认 calibration/ 为权威源后归档删除 biaoding。
- **calibration/ 数据集入 git**：同事把 22 个标定样本文件提交进了 git。将来把
  标定数据加 .gitignore（如 `calibration/datasets/`）防仓库膨胀。
- **datasets/ 运行产物 913MB**（板上磁盘，不入 git）：保留策略待定
  （green_current 480M + 26 个 game 会话）。

## 📊 性能基线（2026-09-24 真机实测，供后续对比）

- 识别+抓取（HOME→GRIP）：约 2-3 秒
- 全流程（到 RETURN_HOME）：约 24 秒
- CAPTURE 内部分布（2026-09-23 口径）：帧等待 17.8% / 帧处理 31.7% /
  YOLO 推理(NPU) 12.4% / stereo_rim 38.1%
- 文档从不承诺端到端耗时数值（只记录 phase_timings_s 指标）
