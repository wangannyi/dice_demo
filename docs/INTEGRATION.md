# 上层应用接入接口

## 1. 边界与调用方式

本仓库提供本机命令行接口，不提供 HTTP 服务。上层应用负责语音、比赛状态、骰子点数识别和胜负判定；本仓库负责抓杯、摇晃、放杯、归位及结果手势。

推荐在 K3 上通过 `subprocess` 调用顶层脚本，固定工作目录为仓库根目录，使用参数数组而非拼接 shell 字符串。Python 内部类属于实现细节，不作为稳定 SDK。相机、标定、TCP、HOME 和动作配置按本安装独立保存。

**同一机械臂只能有一个执行中的任务。** 上层应用应使用全局队列串行执行 Pipeline 和反馈动作，同时禁止 WEB 手动控制。会话锁只保护同一个目录，不是跨目录的全局设备锁。

## 2. 摇骰接口

```bash
DICE_CONFIG="$PWD/configs/green_cup.json" \
DICE_RUN="$PWD/cup_grasp_demo/datasets/app_job_001" \
bash run.sh fast --until place --execute
```

| 参数/环境变量 | 约定 |
| --- | --- |
| 第一个参数 | `step` 人工分步；`control` 上层指令调度；`auto` 连续执行保留常规诊断；`fast` 连续执行精简诊断 |
| `--execute` | 真机执行；缺省仅打印流程，不验证整条硬件通路 |
| `--until ready` | 到抓取位置后停止 |
| `--until grip` | 闭手后停止 |
| `--until shake` | 摇晃后停止，可能仍持杯 |
| `--until place` | 放杯、张手并返回 HOME；顶层入口默认值 |
| `--show` | STEP 中查看图像及 TCP，不建议自动服务使用 |
| `DICE_CONFIG` | 系统配置绝对路径；默认 `configs/green_cup.json` |
| `DICE_RUN` | 会话目录；自动服务建议每次任务使用新目录，避免读取旧状态 |

完整顺序：HOME → CAPTURE → PLAN → APPROACH → GRIP → LIFT → SHAKE → LOWER → OPEN → RETURN_HOME。STEP 输入 `q` 是暂停退出；当前不支持跨进程 `--resume`。不可将重新启动理解为继续上一阶段：重启会从 HOME 张手开始。

STEP 的首次提示前会建立 SDK/CAN 连接、预热相机和加载模型；按 Enter 之间同一进程持续持有这些资源，每次 CAPTURE/TCP 预览仍采集新帧。FAST 也在同一次任务内复用资源。人工分步向 STEP 进程发送 Enter；程序化调度使用下面的 `control` JSON 接口。桌面平面来自标定阶段保存的 `home_table_scene`，运行时只定位本次杯口。

### 常驻阶段控制（供上层集成）

上层需要在阶段之间等待外部事件时使用 `control`。启动后程序**先连接 SDK/CAN、预热相机并加载模型，不发送运动指令**；收到首条 `ready` 事件后再派发命令。整个进程在阶段之间阻塞等待标准输入，不会重新初始化设备。执行阶段仍会按该阶段的逻辑读取新反馈、采集新图像及检查配置，不能把“常驻”理解成动作瞬时完成。

```bash
DICE_CONFIG="$PWD/configs/green_cup.json" \
DICE_RUN="$PWD/cup_grasp_demo/datasets/app_control_001" \
bash run.sh control --execute
```

标准输入、标准输出均为**每行一个 JSON 对象**；阶段诊断输出写到标准错误。每条命令可带唯一 `id`，事件原样回传该 `id`。同一进程内重复 `id` 会被拒绝，避免误重发造成重复动作。命令与事件示例：

| 发送给 stdin | 含义 |
| --- | --- |
| `{"id":"1","command":"status"}` | 读取当前状态与下一阶段；不访问硬件 |
| `{"id":"2","command":"advance"}` | 只执行下一阶段，完成后继续常驻等待 |
| `{"id":"3","command":"advance","until":"GRIP"}` | 顺序执行到 GRIP 并停下；不会跳过中间阶段 |
| `{"id":"4","command":"advance","until":"RETURN_HOME"}` | 完成剩余放杯和归位阶段 |
| `{"id":"5","command":"refresh_perception"}` | CAPTURE 后、APPROACH 前退回 CAPTURE，重新识别和规划；杯位可能变化或计划过期时使用 |
| `{"id":"6","command":"new_cycle"}` | 仅在 RETURN_HOME 完成后重置状态，保留 SDK/相机连接进入下一轮 |
| `{"id":"7","command":"close"}` | 释放 SDK/相机并退出；未完成流程记为 `PAUSED` |

启动时返回 `ready`，每阶段返回 `phase_started`、`phase_completed`，目标阶段结束后返回 `command_completed`。`status` 事件含 `status`、`next_phase`、`completed_phases`；错误命令返回 `rejected`，阶段执行失败返回 `failed` 且进程退出。到 `RETURN_HOME` 后状态为 `COMPLETED`，进程仍等待 `new_cycle` 或 `close`。上层必须持续读取 stdout，按 `id` 和事件判断完成；**不要靠固定睡眠或耗时文本推断动作完成**。阶段执行时追加的命令会排队，按顺序处理；运行中中止仍使用 SIGINT，并核实硬件状态。

```python
import json
import os
from pathlib import Path
import subprocess

root = Path('/home/test2/dice_demo')
session = root / 'cup_grasp_demo/datasets/app_control_001'
session.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, DICE_CONFIG=str(root / 'configs/green_cup.json'),
           DICE_RUN=str(session))
with (session / 'controller.log').open('w') as log:
    proc = subprocess.Popen(['bash', str(root / 'run.sh'), 'control', '--execute'],
                            cwd=root, env=env, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=log,
                            text=True, bufsize=1)

    def receive_until(name, request_id=None):
        while True:
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError(f'控制进程已退出，日志：{session / "controller.log"}')
            event = json.loads(line)
            if event['event'] in ('failed', 'rejected'):
                raise RuntimeError(event)
            if event['event'] == name and (request_id is None or event.get('id') == request_id):
                return event

    def send(command):
        proc.stdin.write(json.dumps(command) + '\n')
        proc.stdin.flush()

    try:
        receive_until('ready')
        send({'id': 'grip-1', 'command': 'advance', 'until': 'GRIP'})
        receive_until('command_completed', 'grip-1')
        # 上层此时可等待语音、游戏规则或人工指令；CAN 和相机仍保持连接。
        send({'id': 'finish-1', 'command': 'advance', 'until': 'RETURN_HOME'})
        receive_until('command_completed', 'finish-1')
        send({'id': 'close-1', 'command': 'close'})
        receive_until('closed', 'close-1')
    finally:
        # 异常时关闭 stdin；正在执行的阶段结束后，进程会收到 EOF 并释放连接。
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass
        return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f'控制进程失败，日志：{session / "controller.log"}')
```

每个 `advance` 都从 `next_phase` 开始顺序运行。若长时间停在 CAPTURE/PLAN 与 APPROACH 之间，杯位可能变化，先发送 `refresh_perception` 再推进到 PLAN/APPROACH。完成 GRIP 后程序不会自行张手或归位；上层应根据当前持杯状态选择后续阶段，不要在异常路径盲目重启 HOME。`control` 仅支持绿杯配置，`--until` 保持默认 `place`；阶段停靠点由 JSON 命令指定。

## 3. 胜负反馈接口

在已放杯、手中无物体后调用。这里的胜负均以**机械臂一方**为准，上层需先把选手角色映射为机械臂/玩家。

```bash
bash run_feedback.sh win --config configs/green_cup.json --session /tmp/dice_feedback_001 --execute
bash run_feedback.sh lose --config configs/green_cup.json --session /tmp/dice_feedback_002 --execute
bash run_feedback.sh draw --config configs/green_cup.json --session /tmp/dice_feedback_003 --execute
```

| 结果 | 别名 | 动作名 | 行为 |
| --- | --- | --- | --- |
| 机械臂赢 | `win` | `yeah` | 举臂并比 V |
| 机械臂输 | `lose` | `thumbs-up` | 举臂并点赞 |
| 平局 | `draw` | `tie` | 到指定姿态，手指两种姿态往返 3 次 |

`--gestures` 指定动作配置，默认 `configs/result_feedback.json`；`--list` 列出动作。不带 `--execute` 仅预览。执行后保持动作姿态，**不自动回 HOME**，也不订阅比赛事件。默认臂速度 50%，臂手同时启动，手使用最大速度指令。

手指六路顺序：拇指尖、拇指根、食指、中指、无名指、小指。七轴角度单位为度。新增、删除动作和调整执行时延见[调试文档](DEBUG.md#9-比大小后的反馈手势)。动作内字段覆盖全局默认值；修改全局速度时注意已有动作也可能配置了覆盖值。

## 4. 状态、返回值和收据

stdout 是人类可读日志，**不要解析耗时行判断成功**。Pipeline 状态文件为 `$DICE_RUN/green_pipeline_state.json`：

| 字段 | 含义 |
| --- | --- |
| `status` | `RUNNING`、`WAITING`（常驻等待命令）、`PAUSED`、`COMPLETED` 或 `FAILED` |
| `active_phase` | 当前/最后进入的阶段，不表示该阶段完成 |
| `events` | 已完成阶段，元素包含 `phase` 和 `status: completed` |
| `phase_timings_s` | 各阶段总耗时，秒 |
| `error` | 失败原因；失败时读取 |
| `receipts` / `recovery_events` | 底层执行记录和恢复记录；具体结构以实际 JSON 为准 |

进程返回值：`0` 正常结束（也包括预览或 STEP 主动暂停）；`2` 配置/执行等已处理错误；`130` 主动中断。其他非零值也按失败处理。配置或依赖初始化失败可能发生在状态文件创建前，不能沿用上一次状态。

完整执行成功需同时满足：进程返回 `0`，本次状态为 `COMPLETED`，且 `events` 含 `RETURN_HOME/completed`。这表示程序完成动作，不代表已经通过视觉或力觉确认持杯，也不代表骰子点数已改变。

反馈动作在 `--session` 下生成 `runs/<时间>_<动作>_<标识>/receipt.json`。成功要求本次进程返回 `0` 且本次收据 `success=true`。`recipe` 保存有效动作参数；手部指令及观察时间完成不等于实测手指到位。细节查同目录请求、实际结果和日志。

## 5. Python 调用示例

以下函数会在调用时执行真机流程。上层串行队列负责调用；`root` 必须是已安装、标定完成的仓库目录。

```python
import json
import os
from pathlib import Path
import subprocess
import uuid


def shake_once(root: Path):
    root = root.resolve()
    session = root / "cup_grasp_demo/datasets" / ("app_" + uuid.uuid4().hex)
    session.mkdir(parents=True)
    env = dict(os.environ, DICE_CONFIG=str(root / "configs/green_cup.json"),
               DICE_RUN=str(session))
    with (session / "application.log").open("w") as log:
        result = subprocess.run(
            ["bash", str(root / "run.sh"), "fast", "--until", "place", "--execute"],
            cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    state_path = session / "green_pipeline_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    finished = any(e.get("phase") == "RETURN_HOME" and e.get("status") == "completed"
                   for e in state.get("events", []))
    if result.returncode != 0 or state.get("status") != "COMPLETED" or not finished:
        raise RuntimeError(f"摇骰未完成，检查 {session}")
    return session, state
```

上层判定结果后，用相同 `subprocess` 方式调用反馈入口，创建独立反馈会话并检查收据。不要把用户输入直接拼成命令或动作文件路径，结果值只接受 `win/lose/draw`。

## 6. 故障和取消

失败后停止派发后续动作，保留日志并读取实际机械臂/持杯状态；不得在异常路径无条件张手或自动重跑 HOME。重复调用不是幂等操作，可能重复抓取或摇晃。

运行中的取消可向前台进程发送 SIGINT。SDK 会按其异常处理尝试停止/保持，但进程退出不是硬件急停确认；上层应等待进程及子进程退出，再允许新任务。不要为达到 UI 超时直接杀进程后立刻派发另一条动作。

上层可以展示任务耗时和“处理中”，不应把固定 0.5 秒、6 秒当成全部状态的完成依据。通信故障、无有效目标、限位及碰撞等仍可能导致中断。

## 7. 配置部署

主配置 `configs/green_cup.json`；摇晃 `configs/joint_shake.json`；反馈 `configs/result_feedback.json`。完整字段与安装见[README](../README.md)，重新布置现场见[标定指南](CALIBRATION.md)。配置变更只在任务结束后进行，新的调用读取新配置。

默认打包命令生成的发布包中，相机标定为参考数据，`installation_requires_calibration=true`。仓库 `main` 和 `--site-active` 包保留当前 K3 的现场配置，只适用于这套固定安装。接收方先完成本机标定与桌面注册，不能直接用开发现场的坐标启动动作。运行记录留在本机，不提交到源码仓库。
