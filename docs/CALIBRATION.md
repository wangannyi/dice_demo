# 标定与相机移动后的校准

## 1. 三种场景

| 场景 | 当前支持方式 |
| --- | --- |
| 首次安装，或机械臂与桌面板相对位置改变 | 手动示教采样 → 手眼求解 → 注册桌面固定板 |
| 希望重复第一次的关节姿态自动采样 | 已保存示教关节；自动运动重采入口尚未实现 |
| 仅相机移动，固定板与基座相对位置不变 | 自动观察固定板 → 恢复相机到基座变换 → 更新 Pipeline 配置 |

固定板不能凭空得到基座坐标。第一次必须通过手眼标定注册它的位置。相机移动后的恢复不需要重复手动摆动机械臂，但目前需要执行下文命令应用结果，不会在每次 Pipeline 中自动替换参数。

## 2. 环境和板配置

```bash
cd /home/test2/dice_demo
source scripts/env.sh
cd nero_calibration
```

需要预览时从 PC 使用 `ssh -Y test2@<K3地址>` 登录。确保 `DISPLAY` 有值，关闭占用相机的 ffplay；不要在标定过程中改变图像参数。

统一采集配置为 1280×720、15 FPS，裁剪 `crop=960:720:220:0`。程序按裁剪更新内参；不要再用 ffmpeg 二次裁剪采样图。手背板配置为 `config/board_hand_redcloth.json`，桌面板为 `config/board_reference_redcloth.json`。

当前板为 4×5 ChArUco。手背板实测总宽 86.5 mm、总高 108 mm；其尺寸修正命令见下文。固定板必须按自身尺寸配置，不要自动套用另一块板的测量值。两块板使用相同标记 ID 时，采手背板需遮挡桌面板，采桌面板需移开或遮挡手背板。全画幅不是限制运动范围的 ROI；如果设置识别 ROI，应只用于区分板子。

## 3. 首次手眼标定

### 3.1 采样

```bash
CAL_RUN="datasets/handeye_$(date +%Y%m%d_%H%M%S)"
./run_k3.sh collect \
  --serial 346222071954 --tcp flange --channel can0 \
  --board config/board_hand_redcloth.json \
  --dataset "$CAL_RUN" --preview
```

手背板需与法兰保持刚性连接。人工调整机械臂，静止后按 Enter 保存，`q` 退出。采样程序只读取反馈，不移动或使能机械臂。至少 12 个有效样本，建议 20–30 个，包含不同位置以及绕不同轴的旋转；只有平移或单轴旋转不足以约束标定。

终端输出角点数和重投影误差。重投影误差小不等于手眼空间误差小。样本拒绝时不会计入有效样本。原目录继续采样：

```bash
./run_k3.sh collect --serial 346222071954 --tcp flange --channel can0 \
  --board config/board_hand_redcloth.json --dataset "$CAL_RUN" --preview --resume
```

### 3.2 求解和实测尺寸修正

普通求解：

```bash
./run_k3.sh solve --dataset "$CAL_RUN" --output "$CAL_RUN/result.json"
```

对当前实测手背板，使用尺寸修正工具重新处理并求解：

```bash
"$CALIB_PYTHON" tools/reprocess_dimensions.py \
  --dataset "$CAL_RUN" --output "$CAL_RUN/dimensions_measured" \
  --width-mm 86.5 --height-mm 108
CALIBRATION="$CAL_RUN/dimensions_measured/result.json"
```

读取 `result.json` 和 `comparison.json` 的质量结果。工具可能写出结果后以非零状态报告质量不通过；文件存在并不代表通过验收。尺寸重处理需与采样数据兼容的 OpenCV 版本。

### 3.3 保存自动采样所需姿态

每个有效样本都会更新数据目录中的 `teaching_poses.json`，记录七轴姿态。请连同样本和结果一起备份。

**当前没有自动回放这些姿态并采样的命令。** 第二次完整手眼标定仍使用 `collect` 人工采样。后续实现回放时，需要从现场起点重新规划路径，不能把旧关节序列直接当成新现场的可执行轨迹。

## 4. 注册桌面固定板

首次手眼完成后，保持相机、基座和桌面板不动：

```bash
REF_RUN="datasets/reference_register_$(date +%Y%m%d_%H%M%S)"
"$CALIB_PYTHON" reference_board.py observe \
  --board config/board_reference_redcloth.json --serial 346222071954 \
  --reference-id table_reference_main --frames 20 --output "$REF_RUN"
"$CALIB_PYTHON" reference_board.py register \
  --calibration "$CALIBRATION" --observation "$REF_RUN/observation.json" \
  --output "$REF_RUN/registration.json"
REGISTRATION="$REF_RUN/registration.json"
```

保存 `registration.json`。它记录固定板到基座的关系，是后续恢复外参的依据。若当前标定质量未通过、经过评估仍决定暂用，在 `register` 命令末尾显式增加 `--allow-provisional`；输出仍保留未通过质量标志，不会变成正式通过。

## 5. 相机移动后的自动校准

条件：板与基座相对位置没变，板尺寸、ID、相机身份及图像配置一致。将 `$REGISTRATION` 设置为首次注册文件的实际路径。

```bash
RESTORE_RUN="datasets/reference_restore_$(date +%Y%m%d_%H%M%S)"
"$CALIB_PYTHON" reference_board.py observe \
  --board config/board_reference_redcloth.json --serial 346222071954 \
  --reference-id table_reference_main --frames 20 --output "$RESTORE_RUN"
"$CALIB_PYTHON" reference_board.py restore \
  --registration "$REGISTRATION" \
  --observation "$RESTORE_RUN/observation.json" \
  --output "$RESTORE_RUN/restored_calibration.json"
CALIBRATION="$RESTORE_RUN/restored_calibration.json"
```

临时质量注册同样需要显式 `--allow-provisional`。相机移动后板可能不再位于原 ROI，应调整 `board_reference_redcloth.json` 的 `image_roi_xyxy`，或遮挡另一块板后使用完整裁剪图像范围 `[0,0,960,720]`。

## 6. 应用标定并更新桌面

停止 Pipeline 后，把结果路径写入顶层配置，再采集该标定下的桌面。以下命令接续上文，在 `nero_calibration` 目录执行：

```bash
CALIBRATION_ABS="$(realpath "$CALIBRATION")"
cd "$DICE_ROOT"
CFG="$DICE_ROOT/configs/green_cup.json"
"$CALIB_PYTHON" - "$CFG" "$CALIBRATION_ABS" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1]); c=json.loads(p.read_text())
c['calibration']=sys.argv[2]
c['green_cup']['installation_requires_calibration']=True
p.write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n')
PY
RUN="$DICE_ROOT/cup_grasp_demo/datasets/green_current"
mkdir -p "$RUN"
"$DICE_ROOT/cup_grasp_demo/calibration_debug/run_planar_shake.sh" table-capture \
  --config "$CFG" --session "$RUN" &&
"$CALIB_PYTHON" scripts/register_home_table.py \
  --config "$CFG" --table-scene "$RUN/planar_table_scene.json"
```

上面两步都成功后，将 `configs/green_cup.json` 中的 `green_cup.installation_requires_calibration` 改为 `false`，然后先执行只读检测和 STEP：

```bash
"$DICE_ROOT/cup_grasp_demo/calibration_debug/run_debug.sh" green-detect \
  --config "$CFG" --session "$RUN" --show
./run.sh step --until ready --show --execute
```

检查杯口定位和 TCP 对应关系后，再运行完整流程。归位桌面缓存与标定结果绑定；相机或桌面变化后，不能继续使用旧缓存。手掌 TCP 是另一项安装几何参数，手眼标定不会自动修正错误的 TCP。
