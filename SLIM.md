# 仓库瘦身与去重清单（SLIM 清单）

> 2026-10-08 全仓冗余盘点产出。与 BUGS.md（缺陷）、TODO.md（功能债）、
> STRUCTURE.md（目录摆放）互补：本清单只管**重复与冗余资产**的收敛。
> 用法与 BUGS.md 相同：报编号即做，做一项划掉一项，完成后在底部记录提交号。

## 🔒 铁律：任何瘦身不得改变交付功能

删除/合并/收敛的每一项，交付行为必须零变化。每项动手时必须按序执行：

1. **改前记锚点**：记下当前提交号（回退基准）；
2. **找齐引用**：全仓 grep 被动目标（含测试 fixture、文档、shell 脚本、
   JSON 内嵌路径），逐个改指向新真源；
3. **板上全量回归**（2026-10-08 实测姿势）：ssh spacemit-k3 后在仓库根执行
   `PYTHONPATH=$PWD:$PWD/third_party/pyAgxArm python3.14 -m pytest tests/ -q`
   （不要 source env.sh——`PYTHONNOUSERSITE=1` 会屏蔽用户 site 里的 pytest；
   旧命令引用的 /home/spacemit/dice-test-deps 已不存在）
   必须回到基线 **689 collected = 659 passed / 30 skipped / 205 subtests，0 失败**
   （2026-10-08 SLIM 收敛后口径：起始 691=661/30/205 实测于 3a872b6，
   L4/P3-15 删 4 个死路径测试 + 补 2 个 preprocess 契约测试后为 689=659；
   x86 开发机不作验收环境：本机 cv2 无 `aruco.detectMarkers`，8 用例必假挂）；
4. **入口冒烟**：视改动面跑 `bash run.sh`（预览模式）、`run.sh control --simulate`、
   `bash calibrate.sh <涉及操作>`；
5. **残留清零**：`git grep <被删名字>` 无输出；
6. 任一步不满足即 `git checkout <锚点>` 回退，重新分析。

## 🎯 新发现（本次盘点新增，按建议顺序）

### L1. ✅（已完成 80e2d23，2026-10-08）字节级双活副本：`calibration/core.py` ≡ `cup_grasp_demo/flow/transforms.py`

**现状**：md5 完全相同（20ba1f37…），两边都在产线运行：
- flow 侧：`planning.py`、`hand_geometry.py`、`flow/grasp.py`、`flow/direct_grasp.py`、
  `flow/debug.py` 以 `cup_grasp_demo.flow.transforms` 导入；
- calibration 侧：`sensors.py`、`reference_board.py`、`calibrate.py`、`auto_collect.py`、
  `tools/reprocess_dimensions.py` 以裸模块 `from core import ...` 导入
  （`calibrate.sh` 直接跑 `python3 calibration/xxx.py`，sys.path[0] 即 calibration/）。

**风险**：改 TCP 变换/手眼解算逻辑时必须记得同步两份，漏一份则标定与抓取
用不同的几何数学，且无任何检查会发现。

**方案**（两步走，第一步零风险）：
1. 先加漂移检测：`scripts/check_source.py` 增加一项 md5 对比，两份不一致即报错；
2. 收敛需先拍板一个问题——**calibration/ 是否要独立拷贝交付**（TODO.md 需拍板区
   的 ../biaoding 双源说明同事有拷贝交付习惯）。若不需要：calibration 各文件改为
   `from cup_grasp_demo.flow.transforms import ...`（calibrate.sh source env.sh，
   PYTHONPATH 含仓库根，导入成立），删除 `calibration/core.py`；
   若需要独立交付：保留双份 + 第 1 步的哈希检查常驻。

**验收**：板上 pytest 基线不变；`bash calibrate.sh` 至少一个只读操作
（如 preview）冒烟通过；`git grep 'from core import'` 清零（方案 b）。

### L2. ✅（已完成 80e2d23，2026-10-08，与 L1 绑定）字节级双活副本：`calibration/image_profile.py` ≡ `cup_grasp_demo/flow/image_profile.py`

**现状**：md5 完全相同（89f8b98c…）。flow 侧被
`vision/capture/realsense_session.py` 导入（产线采集在用）；calibration 侧被
`calibration/sensors.py` 以裸模块导入。

**风险/方案/验收**：同 L1，与其绑定处理（同一个"calibration 独立交付"拍板）。

### L3. ✅（已完成 477aaca，2026-10-08）孤儿测试夹具 1.1MB：`failed_request.json` + `failed_actual.json`

**现状**：`tests/cup_grasp_demo/flow/fixtures/` 下两份摇骰失败回放数据
（659,925 + 439,824 字节，git 跟踪文件体积 TOP1/TOP2）。全仓 `.py/.sh/.md`
**零引用**；git 考古：`b253880`（移除银杯流程与摇晃开发工具）删除了引用它们的
测试，夹具本体遗留至今。

**方案**：`git rm` 两文件。它们是某次真实故障的回放样本，git 历史里永久可查
（`git show b253880^:tests/.../failed_request.json` 可找回）。

**验收**：板上 pytest 基线不变；`git grep failed_request` 清零。

### L4. ✅（已完成 b87593a，2026-10-08，删除根治）urdf 双副本：`models/nero_description.urdf` ≡ `hand_geometry/nero/urdf/nero_description.urdf`

**现状**：md5 相同（d6a5c1cb…，各 9KB）。顶层份是 `kinematics.py` 的默认加载
路径（`load_model()` 无参时）；hand_geometry/ 份被 xacro include 链使用
（`nero_with_revo2_flange_description.xacro` → `nero_description.urdf`，
`hand_geometry.py` 加载该 xacro 链）。

**风险**：改臂网格/惯量参数时漏一份，数值运动学（抓取规划）与 xacro 几何
（手指姿态解算）悄悄分叉。

**方案**：低优先——选定一份为真源（建议 hand_geometry/ 份，xacro 链不好动），
`kinematics.py:442` 的默认路径指向它。注意 xacro 的 include 前缀是
`$(find agx_arm_description)/...`，改链路前先确认解析方式，风险高于收益时可
只做 L1 同款的 md5 漂移检测。

**验收**：`python3 -c "from nero_revo2_control.kinematics import load_model; load_model()"`
成功且结果与改前一致（对比关节零位 TCP 位姿）；板上 pytest 基线不变。

## 📋 既有清单已挂号的冗余（此处仅汇总索引，去原清单修）

**配置副本漂移族**（同一参数多处定义，已量化漂移）：

- **TODO.md #1**：`green_open_cup/config.json` + `stereo_config.json` 与主配置
  `configs/green_cup.json` 漂移（flow/config.json：主配置独有 40 键、共有键
  5 处值不同；stereo_config.json：各 5 处不同——含 confidence 0.25 vs 0.35、
  绑核 [12,13] vs [8,9]、geometry_method、height_mode、放置容差 8 vs 5mm）。
  2026-10-08 实测：主配置 confidence 已改 0.25，stereo_config 仍 0.35——漂移
  正在继续发生。
- **STRUCTURE.md S1**：`flow/joint_test_config.json` 摇骰参数旧副本。
  ✅ 已完成（9b23f61，2026-10-08）。
- **BUGS.md P3-7**：`configs/installation/camera.json` 过期标定副本。
  ✅ 已完成（051386f，2026-10-08）。
- **BUGS.md P2-20**：`fast_finger_duration_s` 四处三个值，strategy 被静默遮蔽。

**死代码族**（零引用或未接线，删除前各自清单内有先修条件）：

- **BUGS.md P3-15**：`vision/inference/yolo_seg.py` 三个解码器 + YoloSegmentor
  死代码（产线走 `cup_perception.decode`）。
  ✅ 已完成（cbbbc6f，2026-10-08，preprocess 保留并补契约测试）。
- **BUGS.md P3-17**：`planar_scene.py` 零引用。
  ✅ 记录已修正（fa69bbc，2026-10-08）——实为活代码（3 处活引用），勿删。
- **BUGS.md P3-16**：`vision/capture/config.py` 的 `calibration_file`/
  `calibration_digest()` 无人调用。
- **BUGS.md P3-4**：`configs/green_cup.json` 的 `shake_study` 死键。
  ✅ 已完成（bd8f8b0，2026-10-08，三份配置同删）。
- **BUGS.md P2-22 / TODO.md**：`model_adapter.py` 未接线且有 bug。

**体积大头**（需拍板区，见 TODO.md）：

- `.git` 99MB（大头是已删除的 Piper 网格与旧模型，需改写历史才能减）；
- `calibration/trajectories/` 3.5MB 标定数据入库（含 3.3MB 的
  teaching_frames.jsonl + 20 个 sample_*.json）；
- 板上 `datasets/` 913MB 运行产物（未跟踪，保留策略待定）；
- `../biaoding` 与 `calibration/` 双源各约 4MB（同事拷贝交付造成）。

## ✅ 审查过、不算冗余（勿重复提议）

- **docs/ 9 篇**：标定入门/进阶、环境、调试、固件、热加载、集成、编码规范、
  lift 容差——主题互不重叠，分层合理。
- **`third_party/pyAgxArm`（248 文件）**：SDK vendor 决策（UPSTREAM_COMMIT 可溯），
  运行必需。**`third_party/wheels/k3-cp314`（5.6MB）**：板上 cp314 装机依赖。
- **STL 网格 21.8MB**（nero + revo2 两手型）：`hand_geometry.py` 手指姿态解算
  的输入数据，两套手型各自完整，不重复。
- **`best_green.q.onnx` 3.6MB**：唯一识别模型，在用。
- **`tests/.../fixtures/controller_limits.json`、`session.json`**：debug.py 与
  多个测试在用。
- **空的重复 `__init__.py`、双 LICENSE**：包结构与上游许可的正常形态。
- **`vendor-site/`（29M）、`vendor-site-deps/`（3.5M）**：本地板端环境，
  已被 .gitignore 覆盖，不占仓库。
- **根目录 3 个 shell 入口**（run / run_feedback / calibrate）：S 清单已审。

## 📝 完成记录

- 2026-10-08 **首轮 SLIM 收敛**（起点 3a872b6，决策：范围 L1-L4+顺手死代码；
  calibration 需独立交付 → L1/L2 双份保留 + 检测为终态）：
  - **L1/L2** 80e2d23：check_source.py 增 DUPLICATE_PAIRS 字节级漂移检测
    （core.py≡transforms.py、image_profile 两对），人为漂移验证报错有效；
  - **L3** 477aaca：删零引用孤儿夹具 1.1MB；
  - **L4** b87593a：urdf 收敛 hand_geometry 单份真源，零位 FK 改前后逐位一致；
  - **P3-15** cbbbc6f：yolo_seg 死代码删除 + preprocess 契约测试；
  - **P3-4** bd8f8b0：三份配置 shake_study 死键删除；
  - **P3-7** 051386f：configs/installation/ 整目录删除；
  - **S1** 9b23f61：joint_test_config 收敛单一真源（debug.py 链同步）；
  - **P3-17** fa69bbc：过期记录修正——planar_scene.py 实为活代码，勿删
    （唯一一次"执行前 grep 复核拦下文档误导"，铁律流程必要性的实证）。
  - 收敛后基线 **689 collected = 659 passed / 30 skipped / 205 subtests**，
    每步板端全量回归通过；git 跟踪体积 36.43→35.36MB（净减 1.07MB，
    18 文件 -894 行；重量大头在历史，见下方"体积大头"区）。
  - **如实记录**：收尾第 4 轮全量跑出现 1 次 failed（未捕获用例名，仅存
    tail 摘要——操作失误），随后全量 3 遍 + 易偶发子集（tests/scripts +
    detection_recheck）5 遍全部复绿，未复现。该次失败时点在纯 .md 改动
    （Step 8/9）之后、所有代码/配置步均已单独回归通过，无因果路径；
    疑为板端既有的时序/IO 偶发。后续若再现，按"捕获用例名→查 BUGS.md
    →立 flaky 条目"处理。
