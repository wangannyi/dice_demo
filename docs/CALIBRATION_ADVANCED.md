# 标定指南

本项目采用眼在手外（eye-to-hand）标定。完整安装包含三项数据：

1. 手背板与机械臂姿态共同求得的 `T_base_camera`。
2. 固定桌面板到机械臂基座的注册关系。
3. 当前标定下的桌面平面。

## 1. 何时执行哪种标定

| 场景 | 操作 |
| --- | --- |
| 首次安装、机械臂基座移动、固定板移动 | 人工手眼标定 → 注册固定板 → 登记桌面 |
| 相机移动，基座与固定板未动 | 观察固定板 → 恢复相机外参 → 登记桌面 |
| 相机、基座和标定板均未动 | 可自动重采手眼数据，用于复核或替换结果 |
| 改变彩色分辨率、裁剪或相机 | 重新人工手眼标定 |
| 只改变帧率 | 可复用空间标定，但应重新验证采集和识别 |

## 2. 准备

在 K3 仓库根目录执行：

```bash
source scripts/env.sh
cd calibration
```

确认：

- 相机、基座和标定板已固定。
- 手背板与法兰刚性连接，运动中不会滑动。
- 桌面固定板在手眼采样时被遮挡，避免同一画面出现重复 marker ID。
- `can0` 已启动，七轴已使能，控制器允许 CAN 控制。
- 预览窗口通过 X11 显示时，使用 `ssh -X` 登录并设置 `QT_X11_NO_MITSHM=1`。

本现场使用 4×5 板，实测四格总宽 86.5 mm、五格总高 108 mm。板配置：

```text
手背板：calibration/config/board_hand_redcloth_cover_fixed.json
固定板：calibration/config/board_reference_redcloth.json
```

相机默认采用 1280×720、6 FPS，裁剪 `[220,0,960,720]`。切换 USB 档位时从仓库根目录执行：

```bash
python3 scripts/set_camera_profile.py usb2
python3 scripts/set_camera_profile.py usb3
```

## 3. 首次人工手眼标定

### 3.1 采样

```bash
CAL_RUN="datasets/handeye_$(date +%Y%m%d_%H%M%S)"

./run_k3.sh collect \
  --serial 346222071954 \
  --tcp flange \
  --channel can0 \
  --board config/board_hand_redcloth_cover_fixed.json \
  --dataset "$CAL_RUN" \
  --preview
```

每个姿态停稳后按 Enter 保存，按 `q` 结束。建议采集至少 15 个姿态，并覆盖画面中心、四周、远近和不同旋转角。每次采样需满足：

- 手背板完整可见，四边留有余量。
- 至少检测到 12 个 ChArUco 角点。
- 机械臂保持静止。
- 姿态与已有样本有明显平移或旋转差异。

采样目录会保存图像、法兰姿态、每帧七轴反馈和相机参数。

### 3.2 画自动标定可见范围

```bash
"$CALIB_PYTHON" auto_collect.py draw-window --dataset "$CAL_RUN"
```

在第一张有效图像上框住手背板允许出现的区域。框应覆盖后续所有采样姿态，并排除画面外无关区域。结果保存在 `$CAL_RUN/board_window.json`。

### 3.3 求解

```bash
./run_k3.sh solve \
  --dataset "$CAL_RUN" \
  --output "$CAL_RUN/result.json"
```

若需要按实测板尺寸重新计算：

```bash
MEASURED_RUN="${CAL_RUN}_measured"
"$CALIB_PYTHON" tools/reprocess_dimensions.py \
  --dataset "$CAL_RUN" \
  --output "$MEASURED_RUN" \
  --width-mm 86.5 \
  --height-mm 108

CALIBRATION="$MEASURED_RUN/result.json"
```

不做尺寸修正时：

```bash
CALIBRATION="$CAL_RUN/result.json"
```

检查结果中的 `quality_passed`、留出样本位置误差和角度误差。质量未通过的结果只应在明确接受误差时配合 `--allow-provisional` 使用；质量标志不会被改写。

## 4. 自动重采手眼数据

自动重采使用人工采样记录的关节轨迹和可见范围。首次采样目录必须包含 `teaching_frames.jsonl`、`board_window.json` 和各样本的逐帧关节记录。

### 4.1 生成计划

```bash
"$CALIB_PYTHON" auto_collect.py plan \
  --dataset "$CAL_RUN" \
  --calibration "$CALIBRATION" \
  --output "$CAL_RUN/auto_plan.json" \
  --capture-only-visibility
```

`--capture-only-visibility` 要求每个停稳采样姿态中的板完整位于框内；两个采样姿态之间的运动过程可以短暂出框。若要求整个运动过程都在框内，删除该参数。

质量未通过但仍决定使用时，显式添加：

```text
--allow-provisional
```

### 4.2 自动运动和采样

`run --execute` 默认先按 `configs/actions/home.json` 的七轴角度回 HOME，确认到位后再从 HOME 进入示教路径；不会改变手指姿态。HOME 与首个采样位之间采用限位检查后的关节插值和平滑速度，HOME 本身不要求手背板可见。

`capture_only` 模式允许这段过渡出框，但采样位仍检查完整板可见。连续可见模式仍对 HOME 到首个采样位执行可见范围检查，出框则在运动前拒绝。回 HOME 或反馈确认失败会中止，不会继续采样。HOME 路径只检查关节限位，不提供桌面、线缆或全臂碰撞保证；首次执行须现场确认当前姿态→HOME→首个采样位的通道畅通。

可用 `--home /绝对路径/home.json` 指定 HOME；`--start-from current` 保留原来的就近示教点起步方式（capture-only 仍要求每轴偏差 ≤2°）。既有计划无需重新生成。记录保存到新采样目录的 `home_start.json`。

```bash
AUTO_RUN="datasets/handeye_auto_$(date +%Y%m%d_%H%M%S)"

"$CALIB_PYTHON" auto_collect.py run \
  --plan "$CAL_RUN/auto_plan.json" \
  --output "$AUTO_RUN" \
  --channel can0 \
  --fps 6 \
  --speed-percent 15 \
  --smooth-speed-deg-s 4 \
  --smooth-acc-deg-s2 6 \
  --show \
  --execute
```

参数含义：

| 参数 | 含义 |
| --- | --- |
| `--speed-percent` | SDK 运动速度百分比，范围 1–100 |
| `--smooth-speed-deg-s` | 平滑轨迹的关节速度上限，°/s |
| `--smooth-acc-deg-s2` | 平滑轨迹的关节加速度上限，°/s² |
| `--fps` | 自动采样帧率，可选 6、15、30 |
| `--show` | 通过 X11 显示每个接受的标定帧 |

动作过快时先降低平滑速度和加速度；动作顿挫时避免只把 `speed-percent` 调得很低，应同时使用连续平滑轨迹。

### 4.3 求解自动采样结果

```bash
"$CALIB_PYTHON" calibrate.py solve \
  --dataset "$AUTO_RUN" \
  --output "$AUTO_RUN/result.json"
```

## 5. 注册桌面固定板

人工手眼标定完成后，露出桌面固定板并保持相机、基座和板不动。

```bash
REF_RUN="datasets/reference_register_$(date +%Y%m%d_%H%M%S)"

"$CALIB_PYTHON" reference_board.py observe \
  --board config/board_reference_redcloth.json \
  --serial 346222071954 \
  --reference-id table_reference_main \
  --frames 20 \
  --output "$REF_RUN"

"$CALIB_PYTHON" reference_board.py register \
  --calibration "$CALIBRATION" \
  --observation "$REF_RUN/observation.json" \
  --output "$REF_RUN/registration.json"
```

临时采用质量未通过的手眼结果时，在 `register` 命令添加 `--allow-provisional`。

`registration.json` 保存固定板到机械臂基座的关系。只要基座和固定板不动，它可用于相机移动后的外参恢复。

## 6. 相机移动后的外参恢复

移动并重新固定相机后，使用自动入口先回 HOME、确认到位，再观察同一块固定板并恢复外参：

```bash
RESTORE_RUN="$PWD/datasets/reference_restore_$(date +%Y%m%d_%H%M%S)"
"$CALIB_PYTHON" reference_board.py restore-auto \
  --registration /绝对路径/registration.json \
  --board config/board_reference_redcloth.json \
  --serial 346222071954 --reference-id table_reference_main \
  --frames 20 --output "$RESTORE_RUN" \
  --channel can0 --speed-percent 15 \
  --smooth-speed-deg-s 4 --smooth-acc-deg-s2 6 --execute
CALIBRATION="$RESTORE_RUN/restored_calibration.json"
```

该入口会移动机械臂到 HOME，但不会修改当前 Pipeline 标定。质量未通过的原注册仍需显式 `--allow-provisional`。HOME 文件可用 `--home` 指定。

若只需独立观察或离线计算，以下旧入口行为不变，不会连接 CAN 或移动机械臂：

```bash
RESTORE_RUN="datasets/reference_restore_$(date +%Y%m%d_%H%M%S)"

"$CALIB_PYTHON" reference_board.py observe \
  --board config/board_reference_redcloth.json \
  --serial 346222071954 \
  --reference-id table_reference_main \
  --frames 20 \
  --output "$RESTORE_RUN"

"$CALIB_PYTHON" reference_board.py restore \
  --registration /绝对路径/registration.json \
  --observation "$RESTORE_RUN/observation.json" \
  --output "$RESTORE_RUN/result.json"

CALIBRATION="$RESTORE_RUN/result.json"
```

固定板身份、几何、相机序列号、分辨率、裁剪或内参不一致时不能恢复，应重新人工标定。

## 7. 应用结果并登记桌面

回到仓库根目录。应用手眼结果会复制文件并将 Pipeline 标记为“等待桌面登记”，不会发送机械臂动作。

```bash
cd "$DICE_ROOT"

"$CALIB_PYTHON" calibration/apply_result.py \
  --result "$CALIBRATION" \
  --config configs/green_cup.json
```

将机械臂移到 HOME，保持桌面无遮挡，然后采集并绑定桌面平面：

```bash
RUN="$DICE_ROOT/cup_grasp_demo/datasets/green_current"

"$DICE_VISION_PYTHON" scripts/table_capture.py \
  --config configs/green_cup.json \
  --session "$RUN"

"$CALIB_PYTHON" scripts/register_home_table.py \
  --config configs/green_cup.json \
  --table-scene "$RUN/planar_table_scene.json"
```

登记成功后 `green_cup.installation_requires_calibration` 自动设为 `false`。先执行不运动预览，再运行真机：

```bash
bash run.sh fast
bash run.sh fast --execute
```

## 8. 输出文件

| 文件 | 用途 |
| --- | --- |
| `configs/calibration/handeye_result.json` | Pipeline 当前使用的相机到基座变换 |
| `registration.json` | 固定板到基座的注册关系 |
| `cup_grasp_demo/flow/green_open_cup/home_table_scene.json` | 当前标定下的桌面平面 |
| `configs/green_cup.json` | 指向上述结果并记录是否需要重新登记 |

标定数据集、图像和运行日志保存在本机，不作为通用安装参数提交。新现场必须生成自己的标定结果。
