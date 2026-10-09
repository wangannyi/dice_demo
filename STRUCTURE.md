# 目录结构优化清单（S 清单）

> 2026-09-29 目录结构审查产出。与 BUGS.md（缺陷）、TODO.md（功能债）互补：
> 本清单只管**文件摆放与仓库卫生**，不含代码缺陷。
> 用法与 BUGS.md 相同：报编号即修，做一项划掉一项，完成后在底部记录提交号。

## 🎯 值得做（按建议修复顺序排列）

### S1. ✅（已完成，2026-10-08）摇骰配置双份已漂移，debug.py 调试走旧参数

**现状**：摇晃动作配置存在两份，内容已漂移：

| 文件 | 引用方 | 状态 |
| --- | --- | --- |
| `configs/actions/joint_shake.json` | 主 pipeline（`configs/green_cup.json:116`） | 活跃，36e4888 / e91aefa 两次调优都落在这份 |
| `cup_grasp_demo/flow/joint_test_config.json` | `flow/debug.py` 默认配置链（`green_open_cup/config.json:147`、`green_open_cup/stereo_config.json:147`） | 无人维护的旧副本 |

**风险**：用 `flow/run_debug.sh`（debug.py）调试时摇晃的是**旧参数**，与真机
pipeline 行为不一致，调出来的结论会骗人。

**修法**：
1. `green_open_cup/config.json` 与 `green_open_cup/stereo_config.json` 的
   `joint_test_config` 值改为 `configs/actions/joint_shake.json`；
2. 删除 `cup_grasp_demo/flow/joint_test_config.json`；
3. 全仓 grep 确认无 `flow/joint_test_config` 残留引用（含测试 fixture）。

**验收**：pytest 全量通过；grep 无残留；`debug.py` 冒烟（preview 模式）不报
配置缺失。

**关联**：TODO.md #1（测试配置漂移收敛）的完整方案会连带重排这两份 json，
本项是其中可独立先做且更急的最小子集——真机行为一致性优先。

### S2. ✅（已完成 3facf37，2026-09-29）`.zcode/` 会话工件解除 git 跟踪

**现状**：`.gitignore` 已含 `.zcode/`，但
`.zcode/plans/plan-sess_7d9db7b4-5d6d-4239-bc6c-dc272f3b7b4b.md`
是先跟踪后 ignore，规则对已跟踪文件不生效，且已随 2026-09-29 推送进远程。

**修法**（3facf37 实录）：`git rm --cached .zcode/plans/plan-sess_*.md`
后提交并推送，`.gitignore` 的 `.zcode/` 规则恢复生效，本地文件保留。
**边界说明**：远程**最新树**已无该文件；但它在历史提交（≤ e91aefa）中
仍存在——按 TODO.md「需拍板」区既有结论，清理历史需改写 git 历史并与
同事拍板，不在本清单范围。

**验收**：`git ls-files .zcode` 输出为空 ✓；推送后远程 hwj_dev 干净 ✓。

### S3. ✅（已完成 0f0a753，2026-09-29）calibration/ 的 10 个测试文件迁入 tests/calibration/

**现状**：`calibration/test_apply_result.py` 等 10 个测试与源码同目录存放，
而 `pytest.ini` 写死 `testpaths = tests`——裸跑 `pytest` 完全不收集它们。
现行基线 482 passed / 30 skipped 只是 `tests/` 的数字。

**风险**：改坏标定代码时常规测试全绿，89 个用例的回归保护形同虚设。

**修法**（0f0a753 实录，与预写方案的差异已标注）：
1. `git mv` 迁至 `tests/calibration/`，测试**保持裸模块导入与裸 patch
   目标不变**（曾试改包式 `from calibration.x import`，因源码内部仍是
   裸 import，patch 的包实例与源码的裸实例分裂导致 11 个用例假失败，
   已回退）；新增 `tests/calibration/conftest.py` 把 calibration/
   源码目录 append 进 sys.path；
2. `test_preview` / `test_calibration` 的 ROOT 锚点改为
   `parents[2] / 'calibration'`，继续解析 `calibration/config/` 板配置；
3. calibration/ 目录下不再有任何 `test_*.py`。

**验收**（板端实测）：迁移前基线 `calibration/` 89 passed；迁移后
`pytest tests/` 收集 **657 = 627 passed / 30 skipped / 201 subtests
passed，0 失败**。x86 开发机不作为验收环境（本机 cv2 无
`aruco.detectMarkers`，8 个用例必挂，与代码无关）。

**板端跑测试姿势**：
`PYTHONPATH=$PWD:/home/spacemit/dice-test-deps:$PWD/vendor-site:$PWD/vendor-site-deps:$PWD/vendor-site/pyAgxArm python3 -m pytest tests/ -q`

## ⏸️ 可选低优先

### S4. ✅（已完成，2026-10-09）`tests/scripts/test_delivery.py` 名不副实

原文件实际测的是 `run.sh` / `run_feedback.sh` 包装器。已改名
`test_run_wrappers.py`（git mv），全仓无模块名残留引用，docstring 同步。

### S5. `cup_grasp_demo/flow/green_open_cup/` 策略数据目录迁移

`configs/green_cup.json` 有 5+ 处指向 `cup_grasp_demo/flow/green_open_cup/*.json`
（grasp_reference、tcp_candidate、CURRENT_GRASP、home_table_scene 等），
属"配置/标定数据住在源码包深处"。理想位置是 configs/ 下，但迁移要动配置、
文档、测试 fixture 与 K3 板上同步，收益/成本比不高。**倾向不动**；仅当
将来执行 TODO.md #1 的策略拆分时顺路迁移到 configs/，勿单独做。

## ✅ 审查过、维持现状（勿重复提议）

- `tests/` 镜像源码目录结构（cup_grasp_demo / nero_revo2_control / scripts /
  vision）——规范，保持。
- `flow/` 40+ 模块扁平堆放——理论可按 perception/planning/execution 分包，
  但 import 面太大、收益存疑，且 TODO.md #1 的配置债更优先，不动。
- `calibration/results/` 标定基线入库——交付可追溯，说得通，保留。
- `.pytest_cache`（pytest 自带 ignore）、`.v2c/` `.vscode/` `vendor-site*`
  ——.gitignore 全覆盖，干净。
- 根目录无散落 `.py`，入口只有 3 个 shell 脚本（run / run_feedback /
  calibrate），干净。

## 📝 完成记录

- 2026-10-09 **S4** 完成：test_delivery.py → test_run_wrappers.py（纯改名，
  docstring 同步，无引用残留）。
- 2026-10-08 **S1** 完成（SLIM 收敛轮）：green_open_cup 两份 json 的
  `joint_test_config` 重指向 `configs/actions/joint_shake.json`，
  删除 `flow/joint_test_config.json` 旧副本；test_delivery 的"双配方独立
  调参"断言随单一真源决策改为单配方校验。debug.py preview 板上冒烟通过，
  pytest 659/30/205 基线一致。
- 2026-09-29 **S2** 完成：3facf37 解除跟踪并已推送（e91aefa..3facf37）。
- 2026-09-29 **S3** 完成：锚点 bf8bc89 → 迁移 0f0a753。测试基线更新为
  **657 collected = 627 passed / 30 skipped / 201 subtests**（板端实测，
  旧基线 482/30 不含标定 89 用例）。
