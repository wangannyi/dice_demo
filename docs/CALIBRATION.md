# 标定与相机移动后的校准

## 1. 三种场景

| 场景 | 当前支持方式 |
| --- | --- |
| 首次安装，或机械臂与桌面板相对位置改变 | 手动示教采样 → 手眼求解 → 注册桌面固定板 |
| 希望重复第一次的轨迹自动采样 | 使用 `auto_collect.py` 的画框、离线规划和自动采集入口 |
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

使用 `--preview` 采样时，`teaching_frames.jsonl` 会记录**每一张预览帧**对应的七轴反馈、时间和手背板识别结果。每个有效样本的 JSON 另记录采集阶段每一帧的七轴反馈。自动重采必须有这份逐帧示教记录；旧数据集只有各姿态的终点角度，不能直接作为自动运动路径。

终端输出角点数和重投影误差。重投影误差小不等于手眼空间误差小。样本拒绝时不会计入有效样本。原目录继续采样：

```bash
./run_k3.sh collect --serial 346222071954 --tcp flange --channel can0 \
  --board config/board_hand_redcloth.json --dataset "$CAL_RUN" --preview --resume
```

### 3.2 画出手背标定板的可见范围

首次保存至少一帧后，按 `q` 暂停采样；暂停期间保持机械臂姿态不变，否则缺少运动轨迹，自动规划会拒绝。仍通过 X11 登录时执行：

```bash
"$CALIB_PYTHON" auto_collect.py draw-window --dataset "$CAL_RUN"
```

在图像上拖出一个**包含所有预定采样姿态下完整手背板**的矩形，按 Enter 保存。选择覆盖预定运动区域的范围，留出画面边缘余量，不要只紧贴第一帧的板。结果保存在 `board_window.json`；继续 `collect --resume --preview` 时，青色框会显示在预览图上。完成手工采样后再次按 `q`。`teaching_poses.json` 仍保存每个有效样本的七轴姿态；连同样本、逐帧轨迹、框和标定结果一起备份。

这里的框是**图像中的板可见范围**，不是机械臂工作空间或碰撞边界。手工示教时应保持完整板始终可见；自动规划会检查记录的每帧和帧间插值是否在框内，运行时再用相机复核。它不能代替桌面、线缆和夹具的避障检查。

### 3.3 求解和实测尺寸修正

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

### 3.4 自动重复采样

首次人工求解完成后，在相同相机内参、裁剪、板安装和机械臂基座条件下运行。若只是相机位置改变，先按照第 5 节通过固定桌面板恢复相机外参，将恢复结果作为这里的 `CALIBRATION`。相机或板尺寸、TCP 改变则不能直接复用旧示教路径。

```bash
AUTO_PLAN="$CAL_RUN/auto_plan.json"
"$CALIB_PYTHON" auto_collect.py plan \
  --dataset "$CAL_RUN" --calibration "$CALIBRATION" \
  --output "$AUTO_PLAN"
AUTO_RUN="datasets/handeye_auto_$(date +%Y%m%d_%H%M%S)"
"$CALIB_PYTHON" auto_collect.py run \
  --plan "$AUTO_PLAN" --output "$AUTO_RUN" \
  --channel can0 --speed-percent 10 --execute
"$CALIB_PYTHON" calibrate.py solve \
  --dataset "$AUTO_RUN" --output "$AUTO_RUN/result.json"
```

如果第一次的手眼结果质量未通过，但你已决定临时采用，在 `plan` 命令末尾显式加 `--allow-provisional`；质量标志仍保持原值。`plan` **不连接硬件**，检查真实示教帧、七轴限位余量以及帧间插值后生成路径。`run` 要求正常 CAN 控制、七轴使能、手背板当前可见且在所画框内，才会从当前位置走向第一个采样位，并逐段运动、逐帧检查、自动保存新的图像和关节反馈。任一帧识别失败、板出框、CAN 异常或关节不到位即停止后续轨迹；已经采到的新样本保留在 `AUTO_RUN`。

自动运动路线跟随首次预览时记录的七轴轨迹，帧间用小关节步长插值；`--max-step-deg` 默认为 2°，`--margin-px` 默认为 8 像素，`run` 默认 10% 速度且上限 20%。自动过程不验证全臂与桌面、线缆的碰撞，**首次自动运行须保持运动区域清空并现场监护**。中途停止的数据集标为 `AUTO_INCOMPLETE.json`，不能直接求解。没有手背板逐帧轨迹的旧数据集需要重新人工采集，不能从仅有的采样终点推断安全过渡路径。

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
