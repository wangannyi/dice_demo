# 仓库瘦身与去重清单（SLIM 清单）

> 2026-10-08 全仓冗余盘点建立；同日首轮收敛完成（L1-L4、S1、P3-15/4/7/17，
> 提交链 21ff3e9..ea4af73，见 git 历史）。本文件只保留**剩余待瘦身项**。

## 🔒 铁律：任何瘦身不得改变交付功能

删除/合并/收敛的每一项，交付行为必须零变化。每项动手时必须按序执行：

1. **改前记锚点**：记下当前提交号（回退基准）；
2. **找齐引用**：全仓 grep 被动目标（含测试 fixture、文档、shell 脚本、
   JSON 内嵌路径）——**文档记录可能过期，以实测 grep 为准**（P3-17 教训：
   记录说零引用、实际 3 处活引用）；
3. **板上全量回归**（2026-10-08 实测姿势）：ssh spacemit-k3 后在仓库根执行
   `PYTHONPATH=$PWD:$PWD/third_party/pyAgxArm python3.14 -m pytest tests/ -q`
   （不要 source env.sh——`PYTHONNOUSERSITE=1` 会屏蔽用户 site 里的 pytest；
   旧命令引用的 /home/spacemit/dice-test-deps 已不存在）
   必须回到基线 **689 collected = 659 passed / 30 skipped / 205 subtests，0 失败**
   （x86 开发机不作验收环境：本机 cv2 无 `aruco.detectMarkers`，8 用例必假挂）；
4. **入口冒烟**：视改动面跑 `bash run.sh`（预览模式）、`run.sh control --simulate`、
   `bash calibrate.sh <涉及操作>`；
5. **残留清零**：`git grep <被删名字>` 无输出；
6. 任一步不满足即 `git checkout <锚点>` 回退，重新分析。

## 🎯 待瘦身（按建议顺序）

### 1. TODO#1 —— 测试配置双副本收敛（当前最大重复项）

**现状**（2026-10-08 复测）：`green_open_cup/config.json` 与
`green_open_cup/stereo_config.json` 仍只被 5 个测试 + debug.py 默认链引用，
与交付主配置 `configs/green_cup.json` 持续漂移：
- config.json：共有键 4 处值不同，主配置独有 41 键，副独有 15 键；
- stereo_config.json：4 处值不同（confidence 0.35 vs 主配置 0.25 等），
  主独有 13，副独有 15；
- debug.py:149 默认仍指旧副本（CLI 调试走旧参数）。

**修法**（TODO.md #1 已有完整方案）：测试改为「加载主配置 + 测试专属覆盖项」
派生，删除两份静态副本，debug.py 默认指向主配置。动手前逐个甄别 5 个测试
文件（direct_approach / green_pipeline / plane_config / green_fast_overhead /
green_image_rim）的断言是"测试专属设定"还是"历史漂移"，前者保留为显式覆盖。

### 2. P2-20 —— `fast_finger_duration_s` 四处三个值

`vision/strategy/green_cup.json`（0.25）被 `configs/green_cup.json`（0.5）
setdefault 静默遮蔽，`green_control.py:392` 兜底 0.25，stereo_config 0.25。
按 README 改 strategy 该键无效且无警告。修法：load_strategy 加同名冲突检测
（不同值 raise）+ 删重复键 + 修 test_green_fast_overhead 的同义反复断言。

### 3. P3-16 —— `vision/capture/config.py` 死字段

`calibration_digest()` 无人调用；README 宣称的"一致性校验"不存在，真正生效
的是 `configs/green_cup.json` 的 `calibration` 键。二选一：接线或删函数改 README
（test_green_capture.py:18 只是把 calibration_file 当必填键构造，不涉及 digest）。

### 4. `model_adapter.py` 未接线（需拍板）

`vision/inference/model_adapter.py` 零引用，但 TODO.md 记为"接新物体时一并做"
的占位，且自身有 bug（BUGS P2-22）。**拍板项**：若确认不再接新物体可删；
否则留待接线时连同 P2-22 修。

### 5. S4 —— test_delivery.py 名不副实

实际测 run.sh/run_feedback.sh 包装器，改名 test_run_wrappers.py（git mv），
确认无别处 import。5 分钟顺手项。

### 6. 数据资产（需拍板，非冗余但有减重空间）

- `calibration/trajectories/` 3.5MB：活跃数据（calibration_workflow.json 引用），
  留 git 有交付可追溯价值；若拍板移出，加 .gitignore 即可。
- 板上磁盘：`datasets/` 运行产物、`runtime/` 138MB（K3 运行时锁，功能性占用），
  均未跟踪，不占仓库。

## ✅ 审查过、不算冗余（勿重复提议）

- **calibration/ 双活副本两对**（core.py≡transforms.py、image_profile）：
  独立交付需要（2026-10-08 用户拍板），**双份保留为终态**，
  check_source.py DUPLICATE_PAIRS 漂移检测站岗。
- **docs/ 9 篇**：主题互不重叠，分层合理。
- **`third_party/pyAgxArm` + `wheels/k3-cp314`**：SDK vendor 决策，运行必需。
- **STL 网格 15.7MB**：两种手型的手指解算输入，各自完整。
- **`best_green.q.onnx`**：唯一现役识别模型。
- **tests fixtures（controller_limits.json、session.json）**：在用。
- **空的 `__init__.py`、双 LICENSE**：包结构与上游许可的正常形态。
- **根目录 3 个 shell 入口**：S 清单已审。

## 📝 完成记录

- 2026-10-08 **首轮 SLIM 收敛**：L1-L4、S1、P3-15/4/7/17 全部完成，
  详见 git 提交链 21ff3e9..ea4af73（逐项验收结论在各自 commit message 与
  历史 SLIM.md 版本中）。基线 689=659/30/205。
