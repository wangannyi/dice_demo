# 目录结构优化清单（S 清单）

> 2026-09-29 目录结构审查产出。与 BUGS.md（缺陷）、TODO.md（功能债）互补：
> 本清单只管**文件摆放与仓库卫生**，不含代码缺陷。
> 用法与 BUGS.md 相同：报编号即修，做一项划掉一项，完成后在底部记录提交号。

## 🎯 值得做（按建议修复顺序排列）

### S1. 摇骰配置双份已漂移，debug.py 调试走旧参数

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

### S2. `.zcode/` 会话工件被 git 跟踪（已推到远程）

**现状**：`.gitignore` 已含 `.zcode/`，但
`.zcode/plans/plan-sess_7d9db7b4-5d6d-4239-bc6c-dc272f3b7b4b.md`
是先跟踪后 ignore，规则对已跟踪文件不生效，且已随 2026-09-29 推送进远程。

**修法**：`git rm --cached .zcode/plans/plan-sess_*.md` 后提交（本地文件保留，
之后被 ignore 兜住）。

**验收**：`git ls-files .zcode` 输出为空；`git status` 干净。

### S3. calibration/ 的 10 个测试文件游离在 pytest 常规收集之外（89 用例）

**现状**：`calibration/test_apply_result.py` 等 10 个测试与源码同目录存放，
而 `pytest.ini` 写死 `testpaths = tests`——裸跑 `pytest` 完全不收集它们。
现行基线 482 passed / 30 skipped 只是 `tests/` 的数字。

**风险**：改坏标定代码时常规测试全绿，89 个用例的回归保护形同虚设。

**修法**：迁移到 `tests/calibration/`，逐项处理：
1. 同目录裸 import（如 `from auto_collect import ...`）改为从 calibration
   导入（calibration 目前无 `__init__.py`，届时决定加包标记或在
   `tests/calibration/conftest.py` 里补 sys.path，以不改动 calibration 源码
   的 import 语义为准）；
2. 测试内相对路径锚点（`Path(__file__).parent` 找 `config/board_*.json`
   等 fixture）改为指向 `calibration/` 的显式路径；
3. 迁移后跑一次全量，新基线数字更新到本文件底部。

**备选**：`pytest.ini` 的 testpaths 加 `calibration`（一行搞定，但破坏
"测试都在 tests/" 的约定，且测试仍与源码混放——不推荐，仅当迁移阻力
超预期时兜底）。

**验收**：裸 `pytest` 收集数 = 原 568 + 89 = 657；全量通过；calibration/
目录下不再有任何 `test_*.py`。

## ⏸️ 可选低优先

### S4. `tests/scripts/test_delivery.py` 名不副实

`scripts/` 里并没有 delivery.py；该文件实际测的是 `run.sh` / `run_feedback.sh`
包装器。改名 `test_run_wrappers.py`（`git mv`），并确认无别处 import 该模块名。
纯改名，5 分钟的事，等顺手时机。

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

（完成后在此追加：日期 + 编号 + 提交号）
