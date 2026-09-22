# dice_demo 目录结构重组计划 v2（hwj_dev 分支，6 步 6 提交）

已确认决策：务实+改名深度；rgb 三桥接模块并入 nero_revo2_control/bridges/；nero_calibration 与 dice_cup_localization 冻结不动；calibration_debug → flow/；**静态动作 JSON 全部集中 configs/actions/**（joint_shake + result_feedback + home），机制零变化只归置路径。

## Step 1：死文件与死引用清理
- 删孤儿 `cup_grasp_demo/control_handoff.py`、过时 `cup_grasp_demo/run_k3.sh`、`nero_revo2_control/offline_plan.py`+其测试、`nero_revo2_control/results/`
- 修 `scripts/package_release.py` 移除已删 kernel_usbcan 白名单（当前打包必炸）
- 10 处 `/home/test2/...` 硬编码默认解释器改 `/usr/bin/python3`（改前 grep tests 确认无断言依赖旧值）

## Step 2：几何资产归一，agx_arm_ros/ 消失
- `agx_arm_urdf/{nero,revo2,LICENSE}` → `nero_revo2_control/models/hand_geometry/`（与 nero_description.urdf 归一处）
- 改 hand_geometry.py DESCRIPTION、debug.py 三处哈希路径、package_release.py mesh 树、check_source.py 跳过表、README/测试提及

## Step 3：桥接模块归位，rgb_hand_tracking/ 消灭 sys.path 裸导入
- 三模块 → `nero_revo2_control/bridges/`（文件名不变 + 空 __init__.py），互导改包名
- 引用方全改正规包导入并删 sys.path 注入：green_sdk_worker（含裸 import nero_revo2_demo 一并包化）、hardware、shake_execution、shake_readback、side_grasp/prepare_hand；shake_cli/planar_shake_cli 哈希路径改 bridges/
- 修测试顺序依赖隐患：test_green_arm_completion/test_green_speed 裸 import 改包导入；test_green_worker_retry 的 mock 键同步
- tests/rgb_hand_tracking/ 3 测试 → tests/nero_revo2_control/bridges/

## Step 4：calibration_debug/ → cup_grasp_demo/flow/（机械替换大头）
- git mv 源目录与 tests 镜像目录
- 两模式全局替换：`cup_grasp_demo.calibration_debug`→`cup_grasp_demo.flow`、`cup_grasp_demo/calibration_debug`→`cup_grasp_demo/flow`（覆盖 .py/.sh/.json/.md；232 处 import + green_cup.json 8 路径字段 + 打包/检查脚本 + docs）
- 完成判据：grep calibration_debug 归零

## Step 5：静态动作库归置 configs/actions/
- `git mv`：configs/joint_shake.json、configs/result_feedback.json → configs/actions/；flow/home_reference.json → configs/actions/home.json（顺手统一命名）
- 引用更新：green_cup.json（joint_test_config、home 字段）、flow 内两份开发 config.json、scripts/result_feedback.py 默认手势路径、tests 断言（test_delivery/test_result_feedback）、README §4/§6 路径
- 明确不变：动作 JSON 的增删机制原样（result_feedback.json 里加动作即生效；摇晃配方、HOME 角度改文件即改动作）；green_cup.json 只留动态抓取（视觉联动）参数

## Step 6：结构自述与收尾
- README §5 新目录树 + 每顶层目录一行职责 + "动作两类"说明（静态=configs/actions/ 的 JSON；动态=green_cup.json 参数+流程代码）
- flow/ 内加简短 README（阶段机/常驻/感知/运动/工具分组）
- package_release.py 全面校对、MANIFEST 重生成、.gitignore 复核

## 验证关卡（每步定向验证，末尾全量四连）
1. 每步：py_compile + 受影响文件定向 pytest
2. Step 2/3/4 后：六入口 import 闭包复跑
3. 最终：全量 pytest 与基线集合对比（25F/133E 零新增）+ check_environment + `./run.sh fast` 干跑 + `control_console --simulate` 冒烟 + 打包脚本 /tmp 冒烟
4. 回退保障：每步单一提交可独立 revert；nero_calibration/dice_cup_localization 冻结，标定数据路径零变更