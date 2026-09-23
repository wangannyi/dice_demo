# 标定与外参恢复

在仓库根目录使用 `bash calibrate.sh <操作>`。脚本自动加载 Python 环境，无需每次 `source`、切换目录或填写相机和速度参数。旧命令仍可用，见 [底层命令说明](CALIBRATION_ADVANCED.md)。

## 常用命令

| 目的 | 命令 | 行为 |
| --- | --- | --- |
| 首次手眼标定 | `bash calibrate.sh first` | 人工示教采样，结束后自动求解 |
| 画允许可见范围 | `bash calibrate.sh window` | 在最近采样数据上画框 |
| 生成自动路线 | `bash calibrate.sh plan` | 使用示教数据和对应结果生成新计划，不运动 |
| 自动手眼标定 | `bash calibrate.sh auto --execute` | HOME → 示教路线采样 → 求解 |
| 注册固定板 | `bash calibrate.sh register` | 拍摄固定板并登记它与基座的关系，不运动 |
| 相机移动后自动校准 | `bash calibrate.sh restore --execute` | HOME → 拍摄固定板 → 恢复外参 |
| 应用结果 | `bash calibrate.sh apply` | 备份 → 应用最近结果 → 采集桌面 → 绑定桌面 |
| 重试桌面登记 | `bash calibrate.sh table` | 不重新应用标定，只重新采集和登记桌面 |
| 重新求解最近数据 | `bash calibrate.sh solve` | 离线求解，保留旧结果 |
| 查看配置和最近输出 | `bash calibrate.sh status` | 只读显示配置与状态 |

任意操作可加 `--dry-run`：只打印将执行的命令，不访问相机、不运动、不写状态。另一个现场使用 `--config /路径/配置.json`。

## 修改参数

统一编辑 `configs/calibration_workflow.json`。相对路径均相对于仓库根目录，支持绝对路径和 `~`。不要编辑运行中的配置；先退出 CONTROL/FAST 和停止网页 Demo，释放相机。

| 配置项 | 用途 |
| --- | --- |
| `serial`、`channel`、`tcp` | 相机序列号、CAN 接口、手眼 TCP（flange/palm） |
| `hand_board`、`reference_board` | 手眼板与固定板的配置文件；尺寸、ROI、分辨率、裁剪在对应板文件中修改 |
| `fps`、`show` | 首次/自动手眼采样帧率、窗口显示；固定板采集帧率使用 reference_board 中的 image_profile |
| `home` | HOME 关节配置 |
| `motion.speed_percent` | SDK 速度百分比，1–100 |
| `motion.speed_deg_s`、`motion.acc_deg_s2` | HOME 和自动采样平滑速度、加速度 |
| `measurement` | 实测棋盘格总宽/高，单位 mm，不含白边；null 表示使用板文件名义尺寸 |
| `allow_provisional` | 默认 false；明确接受误差时设 true，允许未通过质量门槛的结果用于规划、恢复和应用；不会改写质量标志 |
| `capture_only_visibility` | true 表示只要求采样位完整可见，途中可出框 |
| `max_step_deg`、`margin_px` | 生成计划时的关节插值步长与图像边距 |
| `reference_id`、`reference_frames` | 固定板身份和采样帧数 |
| `teaching_dataset`、`teaching_calibration`、`plan` | 显式指定示教数据、对应标定和自动计划 |
| `registration`、`dataset`、`result` | 显式指定固定板登记、待求解数据和待应用结果 |
| `output_root`、`state_file` | 新数据目录和最近输出记录 |
| `pipeline_config` | 应用结果时更新的抓取配置 |

路径选择项填 `null` 时使用自动记录的最近结果；非 null 值优先于状态。新输出使用时间戳和随机后缀，不覆盖旧采样。复制配置用于另一台机器时，同时修改 state_file 和 output_root，避免混用状态。

当前 K3 沿用已经测试的速度 40% / 20°/s / 20°/s²，实测板尺寸 86.5 × 108 mm。首次换场地应根据实际硬件和板尺寸重新配置。自动手眼结果与固定板恢复结果都不会自动覆盖抓取标定；使用 `apply` 才会应用。

## 首次安装

固定相机与机械臂基座，将 ChArUco 手眼板刚性固定在末端，遮住桌面固定板。手背单个方形码不是手眼板。两块相同编码的板同时可见可能误识别；现有检测不会证明板已刚性安装。首次采样不驱动机械臂回 HOME，保持原人工示教方式。

```bash
bash calibrate.sh first
```

每个姿态停稳后 Enter 保存，q 结束并求解。建议至少 15–20 个姿态，覆盖平移和多个旋转方向。失败或 Ctrl+C 的采样不会更新“最近完整数据”；日志中的目录保留，可将 dataset 设置为该目录后单独 solve。首次采样成功会更新 teaching_dataset、teaching_calibration；若配置里有旧路径，改回 null 才使用新记录。

```bash
bash calibrate.sh window
bash calibrate.sh plan
```

画框覆盖全部示教采样位。修改速度无需重新生成计划；修改示教数据、标定或可见框后必须重新 plan。

接着移除手眼板遮挡，露出固定桌面板（基座与固定板不能移动），登记固定板：

```bash
bash calibrate.sh register
```

机械臂放到 HOME，确保红布桌面充分可见，再应用并登记桌面：

```bash
bash calibrate.sh apply
```

## 仓库内置示教轨迹

仓库包含 2026-09-22 示教的 20 个自动标定姿态：

```text
calibration/trajectories/handeye_auto_teach_covered_20260922
```

`configs/calibration_workflow.json` 已将 `teaching_dataset` 和 `teaching_calibration` 指向该目录，并设置 `max_step_deg` 为 10。首次使用或轨迹内容变化后生成新的机器本地计划：

```bash
bash calibrate.sh plan
```

命令应生成 20 个采样目标和 181 个路径点，输出计划路径写入工作流状态。不要复制旧计划文件；计划绑定源文件绝对路径和哈希。轨迹不含采样图片，只能在机械臂基座、D435i、手背板安装以及 1280×720、裁剪 `[220,0,960,720]` 均未改变的同一套设备上复用。内置参考结果的质量标志为 false，配置中的 `allow_provisional=true` 表示采用现场已接受的误差，不会改写质量标志。

## 自动标定

沿用同一安装、示教路线和手眼板安装关系。遮住固定板，确认当前位置→HOME→首采样位无障碍。HOME 过渡仅检查关节限位，不是完整碰撞规划。

```bash
bash calibrate.sh auto --execute
bash calibrate.sh apply
```

在两个命令之间拆下末端手眼板，将机械臂放到 HOME 并露出红布桌面；apply 本身不移动机械臂。自动采样结束不会自动回 HOME。自动标定更新手眼外参后，如果需要把它作为以后固定板恢复的基准，请重新 register。

窗口在采样停稳时更新，平滑运动时不处理 X11 事件。Ctrl+C 中断后保留数据和不完整标记，不继续求解、应用。

## 相机移动后的自动校准

必须已运行 register；基座、桌面固定板不能移动，reference_id 和板尺寸必须保持一致。露出固定板；相机视角改变后检查固定板 ROI。

```bash
bash calibrate.sh restore --execute
bash calibrate.sh apply
```

恢复先回 HOME，再观测固定板。缺少登记文件会报错，不会用手眼结果冒充固定板登记。恢复结果继承原标定的质量状态。

## 备份、失败恢复和验证

apply 先保存 pipeline_config.json、handeye_result.json 和 home_table_scene.json 到新的 backup 目录，status 显示其路径。然后调用原有应用工具，将抓取流程设为“等待桌面登记”；只有桌面采集与绑定成功才解除该状态。

如果桌面采集失败，解决相机占用或桌面遮挡后执行 `bash calibrate.sh table`。不要只改哈希或手动清除等待标记。

回滚时先停止抓取进程，将同一次备份的三个文件分别恢复到 pipeline_config 指定的配置文件、原 calibration 文件和原 home_table_scene 文件。三者必须一起恢复，再重启会话。

完成后先预览，再由现场人员启动实际抓取：

```bash
bash run.sh fast
bash run.sh fast --execute
```

求解默认沿用 5 mm / 2° 质量门槛。角点重投影误差低不代表手眼精度通过。新入口的离线测试不等于真机动作验证。
