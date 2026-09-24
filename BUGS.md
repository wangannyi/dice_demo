# Bug 清单（2026-09-24 全项目深挖）

> 4 路并行审计产出（常驻控制 / 运动执行 / 视觉感知 / 入口配置桥接），P1 结论已逐条人工复核。
> 基线：HEAD 5d6f4c1，全量测试 464 passed / 30 skipped 全绿、compileall 零错误——
> **以下所有 bug 都在现有测试盲区里**。修一条划一条；与本清单无关的优化项见 TODO.md。
>
> 行号以 5d6f4c1 为准，改动后可能漂移，以描述定位为准。

---

## 🔴 P1 —— 必须修（3 条，均已亲自验证坐实）

### [ ] P1-1 失败保命链第四处 0.1s 硬编码：`fresh_js_hold`

- **位置**：`cup_grasp_demo/flow/joint_delivery.py:307`（裸调 `core.fresh_feedback(session)`）
- **调用点**：`joint_execution.py:406`、`shake_execution.py:424`
- **问题**：5d6f4c1 修了同链下游的 verify_stop，但上游的 `fresh_js_hold` 仍走默认 0.1s。
  同一条失败链两半用不同预算：hold 取样 0.1 / hold 验证 0.3，先炸的必然是严格的那半。
- **板上实证**（`cup_grasp_demo/datasets/game_20260924_163812_c882ba06/runs/...163854_green_joint_shake_d47d58/actual.json`）：
  ```
  error      = RuntimeError: Joint feedback exceeds 100 ms freshness limit
  hold_error = Joint feedback exceeds 100 ms freshness limit   ← 就是 fresh_js_hold
  restored_speed_percent = None    ← hold_verified 拿不到 → 速度不恢复
  ```
- **连锁**：`joint_execution.py:418-422` 要求 `hold_verified` 才恢复 `restore_speed_percent`，
  机械臂最终停在"没人验证过、也没降速"的状态。
- **修法**：`fresh_js_hold(core, session, limits, *, joint_max_age_s=.1)` 加形参，
  两个调用点传 `freshness_limit`。hold 的安全性来自 `state_ok` + `validate_target`，
  不是 packet 年龄——放宽取样预算不降低安全等级。

### [ ] P1-2 rounds 连跑谎报局数：`already_completed` 空转轮也计数

- **位置**：`cup_grasp_demo/flow/green_control.py:140-143`（无条件 `completed_rounds += 1`）
  配合 `_advance_single:156-158`（目标已超越时 `_reject(already_completed)` 却 `return True`）
- **问题**：`rounds>1` 且目标非 `RETURN_HOME` 时，第 2..N 轮只发拒绝事件不执行任何阶段，
  最后仍发 `rounds_completed rounds=3, runs=0`（自相矛盾），无 `run_completed`。
  **控制台菜单 83 行自己推荐的 `{"command":"advance","rounds":3}` 就会踩**
  （无 until → target_index=next_index=0，只跑 HOME）。
- **危害**：游戏程序按 rounds 协议接入会把"没跑"当"跑了 N 局"。
- **修法**：`_handle` 的 advance 分支加 `rounds > 1 and target_index != len(PHASES)-1 →
  rejected(invalid_rounds)`；同时让被 `already_completed` 拒绝的轮次不计入
  `completed_rounds`（或在 `_advance` 入口先判 `next_index > target_index` 拒绝整个连跑）。
- **测试补**：① `rounds>1` + `until=GRIP` 被拒；② 无 until + rounds>1 被拒（当前菜单路径）。

### [ ] P1-3 `register_home_table.py` 把绝对路径写回 git 跟踪的配置

- **位置**：`scripts/register_home_table.py:34,45-46`
- **问题**：`register()` 拿 `planar_scene.verify()` 的返回值整份回写，而 `verify` 返回
  `load_config()` 的**已锚定视图**（`core.py:111-113` 把 home/calibration/
  orientation_reference/tcp_candidate/grasp_config 五个键无条件改成 ROOT 绝对路径）。
  按文档跑一次桌面登记 = 把 `/home/...` 写进 `configs/green_cup.json`。
- **背景**：之前修的"同事绝对路径"问题只修了读侧（load_config 锚定容忍），写侧仍在生产。
- **连带**：`package_release.py:37/40` 的 `relative_to(ROOT)`/`is_relative_to` 换机打包
  直接 `ValueError`；出厂记录已见 `/home/test2/dice_demo_hwj/...` 残留 3 处
  （`green_open_cup/home_table_scene.json:23`、`configs/calibration/handeye_result.json:262`、
  `configs/installation/camera.json:262`）。
- **修法**：`register()` 只改原始 JSON 的目标键——`raw = json.loads(config.read_text())`
  后改 `raw['green_cup']['installation_requires_calibration']` 再 `atomic_json(config, raw)`，
  不回写锚定视图。
- **测试补**：不 mock `verify` 的用例，断言登记后 5 个路径键仍为相对路径。
  （现有 `tests/scripts/test_register_home_table.py:38-39` 用 patch 绕过了真实 load_config，零覆盖。）

---

## 🟠 P2 —— 真实缺陷，特定条件触发

### A. SDK 自愈与连跑（最近两个新功能各有硬伤）

#### [ ] P2-1 零发送重发分支实际永远走不进去（两个独立死因）

- **位置**：`green_runtime.py:75-82`（`_zero_tx_failure`）、`:125`（重发）
- **死因 1**：硬件回执只有 `tx_frames` 键（`hardware.py:220`），`_zero_tx_failure` 读的是
  `tx` → `tx.get('actual_tx_count')` 恒 None → 判定恒 False。于是 snapshot/run 的零发送
  死亡必然走 `close()+raise`，承诺的重发永不发生。
- **死因 2**：即便判定通过（只有 shake 回执带 `tx` 块），重发 `self._call_once(command,
  output, request)` 复用同一 output——而 worker 第一件事就是 `open('x')` 创建 `.log`
  （已被第一次尝试创建）→ 立即 FileExistsError，第二个 worker 也死；且该异常在 try 之外
  抛出，绕过全部失败分级，无 close、日志句柄不释放。
- **测试为何没抓到**：`test_sdk_restart.py` 的假 worker 不创建 `.log`、不检查
  `output.exists()`、直接覆盖 output，还伪造了 `run/snapshot` 根本不产出的 `tx` 块——
  在真机永不出现的场景下通过。
- **修法**：`_zero_tx_failure` 兼容 `tx_frames == []`；重发用新 output
  （如 `output.with_name(stem+'_retry')`）或先清理旧 receipt/log；重发包进同一套分级；
  假 worker 复刻 `.log`/`output.exists()` 语义。
- **附带**：`_zero_tx_failure` 漏检 `parameter_write_commands_sent`，而 worker 自己的闸门
  `retryable_shake_start`（`green_sdk_worker.py:18-26`）要求它为 0——客户端拿着比 worker
  更弱的证据授权重发。

#### [ ] P2-2 重发路径丢 `on_dispatched`（被 P2-1 掩盖，修好即现形）

- **位置**：`green_runtime.py:125` vs `green_pipeline.py:366-371`
- **问题**：GRIP 的 `run` 调用带 `on_dispatched=self.launch_following`，重发的
  `self._call_once(command, output, request)` 没传 → `_route_future` 永不决议 →
  LIFT 阶段 `finish_following()` 的 `future.result(timeout=120)` 白等满 120s 后
  TimeoutError。
- **修法**：重发透传 `on_dispatched`（注意只在第一次真正 dispatch 时执行一次）。

#### [ ] P2-3 局间 stop 有 TextIOWrapper 缓冲盲区

- **位置**：`green_control.py:62-84`（`_inter_round_request` 用 select 探 fd）、`:106-111`
- **问题**：`serve()` 用 `readline()`（8KB readahead），advance 与 stop 同批写入时，
  第一条 readline 把后续行吞进用户态缓冲 → fd 已空 → select not ready → peek None →
  跑完全部轮。跑完后 stop 落到 `_handle` 被拒 `not_in_multi_round`。
- **触发**：任何同一次 flush 批量写 advance+stop 的管道客户端（游戏程序典型写法）。
- **测试为何没抓到**：`test_green_control.py` 的 `_pipe_session` 写完即关 writer，
  EOF 使 select 恒 ready——测试通过的理由与实际语义无关。
- **修法**：peek 前先问缓冲（自维护 pending 行队列，或改 `os.read` 裸读/selectors）；
  测试改为 writer 保持打开、延迟写入。

#### [ ] P2-4 多轮期间 state 丢 `rounds_total`；失败时轮次字段残留

- **位置**：`green_control.py:135-139`（只写 remaining）、`:140-143`、`:196-206`
- **问题**：每轮完成 `_reset_for_next_run()` 后 `rounds_total=None`，第 2 轮只补
  remaining → 中途 status 永远 `rounds_total=null` + `rounds_remaining=2`；
  轮内失败时两个字段从不清理，FAILED state 里残留 `rounds_total=3`。
- **修法**：`rounds>1` 分支每次同时写两个字段；失败/停止路径统一清理。

### B. 反馈新鲜度病根剩余接线（3559c19/d3a1634/5d6f4c1 的同族）

#### [ ] P2-5 状态流被内层写死的 0.25 架空

- **位置**：`joint_execution.py:325`（`state_max_age_s=max(.25, freshness)` 名义可放宽）
  vs `nero_revo2_control/bridges/visual_servo_probe.py:219/235/249`（三处写死 `.25`：
  arm status / 7 轴 enable / 二次刷新判据）
- **问题**：freshness=0.3/0.5 时 state_max_age_s 的放宽永不生效，内层先抛
  `No fresh arm status feedback` / `No fresh joint enable feedback`。
  这是测试注释里刚打过的补丁（100ms 那次）在 0.25 上的同构复发。
- **修法**：`fresh_feedback` 加 `state_max_age_s=.25` 形参替换三处字面量，
  `FeedbackReader._run` 透传 `self.state_max_age_s`。

#### [ ] P2-6 手指反馈 0.25 不可配，与臂侧不对称；另有第三套 1.0s 门限

- **位置**：`finger_feedback_probe.py:146`（写死 `.25`）；调用方
  `shake_execution.py:72-77`（`闭手位置反馈不新鲜，停止摇晃`）、`:384-387`；
  第三套：`grasp_execution.py:43/70`（写死 `<= 1`）
- **问题**：臂侧已放宽 0.3，手侧仍硬编码 0.25。摇骰中每 0.1s 查一次手形，CAN 被
  100Hz 关节流占满时手包松 >250ms 即在**运动进行中**中止（比启动前中止危险），
  随后落进 P1-1 的 0.1s hold 链。同一项目手指侧 0.25/1.0 两套写死门限、零配置项。
- **修法**：`_copy_getter(..., max_age_s)`（或 `hand_feedback(..., max_age_s=)`）从
  `request['feedback_freshness_limit_s']` 取值（手/臂同总线，同一预算最自然）；
  `grasp_execution` 的 1.0 一并收口。

#### [ ] P2-7 `PassivePoseSession.snapshot()` 硬顶 0.25s：配置 0.1..0.5 名不副实

- **位置**：`passive_pose_bridge.py:227-230`（`0 <= age <= .25` 否则重试至 2s 超时）、
  `:197`（`start()` 自报 `freshness_limit_s: .25`）
- **问题**：对外配置校验允许到 0.5（joint_execution/joint_delivery/green_pipeline 三处），
  实际有效上限永远 0.25。现场调到 0.3 以上且真出现 0.25~0.3s 调度间隔时，报的是
  `No fresh stable four-packet snapshot`（与 freshness limit 无关），无法分辨是配置
  太小、总线坏还是接线漏了。当前 0.3 能工作只因实测最大年龄 0.129s 未触雷。
- **修法**：`.25` 与 `.02` 提为 `PassivePoseSession(deadline_s, max_age_s=..., max_span_s=...)`
  由 `fresh_feedback(joint_max_age_s=...)` 透传；或把配置范围收敛到 `.1..25` 并在
  start()/文档写明。

#### [ ] P2-8 测量路径写死 0.1s：0.3 配置下 16% 合格样本被静默丢弃

- **位置**：`joint_profile.py:366-372`（`0 <= observed - stamp <= 0.1` 否则 `continue`）
- **问题**：`measurements()` 用它算 `unique_feedback_samples`/频域周期/
  `tracking_verified`。配置 0.3 时执行侧接受的 0.1~0.3s 行在这里被丢——
  实测回执 192 行中 31 行（16%）会被丢，样点变稀，最坏把 `tracking_verified`
  从真判成假（执行与验收结论不一致）。
- **修法**：`packet_samples`/`measurements` 加 `max_age_s` 形参，由 run 传
  `freshness_limit`。

#### [ ] P2-9 起点静止检查 0.1s + 失败无 failure_code（⚠️ 需拍板）

- **位置**：`joint_execution.py:122-123,157`；`visual_servo_probe.py:286,331`
- **现状**：5d6f4c1 **有意保持严格**（注释：安全判定点用陈旧反馈会假阳性）。
- **反方论点**：`persistent_stopped_window` 的设计目的就是容忍 LIFT 尾部错拍
  （docstring 自己写了 "Two packets can therefore straddle the tail of the lift"），
  却用 100ms 拒收；且失败不带 `failure_code`，走不了 green_pipeline.py:902 已有的
  `start_position_changed` 重试路径，整轮直接硬失败。
- **实测背景**：同一会话 192 个反馈样本中 31 个（16%）packet age > 0.1s，最大 0.129s
  ——LIFT 刚结束正是最拥挤的时刻。
- **两个选项**：① 透传 freshness_limit（stationarity 是可行性判定不是安全门）；
  ② 保持 0.1 但把该异常映射成 `failure_code='start_position_changed'` 走既有
  no-motion replan 恢复。**建议 ②（不降安全等级，补恢复路径）**。

### C. 摇骰热路径与清理

#### [ ] P2-10 ShakeGuard 每读两次全量 deepcopy（自激新鲜度失败）

- **位置**：`visual_servo_probe.py:107`（`report()` 里 `deepcopy(self.history)`）、
  `:255-256`（每行反馈调两次）；shake 侧未像 joint 侧换 `JointGuard`
- **问题**：100Hz 反馈循环 = 200 次/秒全量深拷贝，history 随运行时长线性增长。
  实测 1462 行时单次 deepcopy 4.89ms ≈ 98% 单核；30s 摇骰膨胀到 1.5 万行、
  单次 ~50ms——循环直接崩。读反馈被拖慢 → packet age 抬高 → 正好触发新鲜度失败。
- **修法**：给 `AuditedSendGuard` 增加 `counters()`（只取 tx_attempts/
  actual_tx_count/uncertain，不 deepcopy），`fresh_feedback` 只调一次；
  或 shake 复用 JointGuard 的增量实现。

#### [ ] P2-11 清理期 `session.close()` 未包异常：shake 回执不落盘 / joint 审计失真

- **位置**：`shake_execution.py:450`、`joint_execution.py:431/474-479`
- **问题**：`robot.disconnect()` 一抛（USB-CAN 掉线），shake 进程带栈退出、
  `args.output` 永不写——上层 `read_json(actual.json)` 只看到"文件不存在"而非
  真实原因；joint 的兜底 stub 把已发运动记成 `motion_attempted=False`。
  常驻模式下 run() 异常还会直接杀死 green_sdk_worker 的 for 循环
  （`green_sdk_worker.py:109-110` 无 try）。
- **修法**：照 vsp 写法 `try: session.close() except BaseException as exc:
  report['disconnect_error']=...`；joint 的 main 不要用 stub 覆盖整份报告。

#### [ ] P2-12 `FeedbackReader.close()` join 超时后抛异常，读线程可能与 hold/verify_stop 并发

- **位置**：`joint_stream.py:92-97`（join 5s 后 raise）；调用点 `joint_execution.py:385`（finally 里）
- **问题**：单次读最坏 ~18s（snapshot 2s + status 2s + enable 2s × 递归 3 轮），且
  snapshot 内部 sleep 用 session 自己的 time.sleep，stop 事件打不断。close 抛出后
  读线程仍活着继续调 `session.snapshot()`，主线程开始 hold/verify_stop → 两线程并发
  读写 `PassivePoseSession` 共享状态 → 随机爆 `not all four joint packets advanced`；
  随后 `session.close()` 断开 SDK，daemon 线程还在操作已断开的 SDK。
- **修法**：close 不抛（记 event / 返回 bool）；snapshot 的 sleep 换成可注入以支持打断。

### D. 入口与工具

#### [ ] P2-13 `action_registry` 拒载网太窄：手误 json 直接崩主入口

- **位置**：`scripts/action_registry.py:62-91`
- **问题**：动作体非 dict（`"yeah": null`）→ `recipe_for` 抛 `AttributeError`；
  `aliases` 非 dict（`"aliases": 5`）→ `_group_names` 的 `set(5)` 抛 `TypeError`，
  且在合并循环 try 之外。两者都越过"整组拒载、不影响他组"的契约：
  控制台主入口（`control_console.py:258` 的 load_registry 在 try 外）直接 traceback；
  常驻侧 build_action_runtime 被 except 兜住但结果是**全部手势静默失效**。
- **修法**：`_first_recipe_error` 改捕 `Exception` 并把异常原文写进错误消息；
  load_registry 增加 aliases 类型检查整组拒载；补负例测试（recipe 为
  null/number/list、aliases 为 number）。

#### [ ] P2-14 菜单承诺的 `9` 从未实现

- **位置**：`scripts/control_console.py:83`（菜单文案）vs `:90-101`（BASE_CHOICES 无 9）
  且 `9` 被 `RESERVED_KEYS` 占死
- **问题**：操作员按 9 只得"无效指令"，以为"机械臂进程已释放"，实际常驻进程继续占着
  CAN/相机。唯一等价操作是 Ctrl-D，菜单没写。
- **修法**：二选一——加 `"9"` 本地退出分支（关 stdin 不发 close），或菜单改写
  "Ctrl-D 仅退控制台"。

#### [ ] P2-15 Ctrl-C 提示语与事实不符；wait 无超时

- **位置**：`control_console.py:339-346`；根因 `:157-160`（Popen 未 `start_new_session`）
- **问题**：终端 Ctrl-C 同时投递给子进程 → `_advance_single` 的 perform 被
  KeyboardInterrupt 打断 → 状态记 **FAILED**（不是提示语说的"阶段结束后释放、PAUSED"）；
  `backend.wait()` 无超时，子进程卡在长 SDK 调用/采图时控制台永久挂住。
- **修法**：`Popen(..., start_new_session=True)` 让 Ctrl-C 只作用于控制台，走
  "close_stdin → 阶段结束后 EOF → PAUSED" 正规路径；wait 加超时并提示
  "子进程未在 N 秒内退出，硬件状态未知"。

#### [ ] P2-16 `visual_servo_probe` 无 SIGTERM 处理；Ctrl-C 中止路径会发运动指令

- **位置**：`visual_servo_probe.py:446-455`、`:344-356`、`:481-512`
- **问题**：① `kill <pid>` 绕过 finally——不 close、不写 output、guard 不恢复，
  机械臂停在最后一条命令上。② `except BaseException` 把 Ctrl-C 也当"可复位失败"，
  中止时反而触发 `fresh_hold()` 里的 `move_j(target)`（用半途采样 q_rad 当目标）。
  ③ `hold_verified` 全仓库无处置 True，字段名承诺了不存在的验证。
- **修法**：像 finger_feedback_probe 一样把 SIGTERM 转成异常；KeyboardInterrupt 走
  "只 guard.allowed=False + close"，不发 hold；`hold_verified` 改名
  `hold_requested_only` 或补一次复验。

#### [ ] P2-17 `finger_feedback_probe.main` 失败路径 `UnboundLocalError`

- **位置**：`finger_feedback_probe.py:278-291`
- **问题**：`report` 只在 try 体内赋值，finally 无 except——`runtime_loader()`
  （import can 失败等）或 probe() 抛异常时 ：289 引用未绑定变量，输出契约
  （每次调用都有 JSON 报告）在错误路径上是破的。对比姊妹工具
  `visual_servo_probe.main:496-503` 有正确写法。
- **修法**：包 `try/except BaseException`，失败时构造
  `{'schema':1,'kind':...,'valid':False,'blocker':...}`。

### E. 视觉/采集

#### [ ] P2-18 CLI 采集 `fresh=False`：丢帧开关失效且元数据谎报

- **位置**：`vision/capture/realsense_session.py:175,201-215,288,305-312`
- **问题**：`capture()` 的 `fresh` 默认 False，`main()` 从不传 True——CLI 入口
  （debug.py 的 capture_rgbd、green_pipeline._capture_once 非 persistent 分支）
  预热 20 帧/丢弃 0 帧，而元数据仍写 `fresh_discard_frames=2`。D435i 慢流会重复
  返回同一 frameset（代码注释自认），批次可能含重复物理帧，`nanmedian` 静默按
  重复样本平均；立体分支会抛 `Repeated stereo frame pair`。
- **修法**：`main()` 传 `fresh=True`（或 count>1 强制 fresh）；元数据记录实际执行的
  丢弃次数。

#### [ ] P2-19 相机 `close()` 先清 `started` 再 stop：stop 失败即永久放弃设备

- **位置**：`realsense_session.py:292-302`、`:189`
- **问题**：`started=False` 在 `pipeline.stop()` 之前——stop 一抛（USB 掉线），
  该 session 永远处于"未启动"，再次 close 直接跳过 stop，设备保持占用——
  **"下次打不开相机"的典型路径**。且 join 只等 5s，在途采集预算 30+5*count 秒，
  读帧线程还在 wait_for_frames 时主线程就并发 stop。
- **修法**：try/finally 保证 started 只在 stop 成功后清（幂等）；join 超时后不并发
  stop，等线程自然退出。

#### [ ] P2-20 strategy 合并被主配置静默遮蔽：`fast_finger_duration_s` 四处三个值

- **位置**：`green_pipeline.py:62-66,195-198`（`g.setdefault(key, value)`）；
  `vision/strategy/green_cup.json:36`（0.25）vs `configs/green_cup.json:230`（0.5，生效）
  vs `green_control.py:392`（兜底 0.25）vs `green_open_cup/stereo_config.json:245`（0.25）
- **问题**：`load_strategy` 只校验 4 个必需键不检测同名冲突；唯一同名键恰好冲突，
  setdefault 让 strategy 的 0.25 永不生效。按 README 指导改 strategy 该键无效且无警告。
- **修法**：`load_strategy` 加冲突检测（同名不同值 raise）或明确优先级写进 README；
  删重复键之一。`test_green_fast_overhead.py:17` 的断言是同义反复
  （`g[k]==g.get(k,g[k])`），换成具体数值断言。

#### [ ] P2-21 `realsense_session.py` 路径基准 off-by-one

- **位置**：`realsense_session.py:12-17`（`parents[1]` = vision/，应为 `parents[2]` = 项目根）
- **问题**：未 source env.sh 时独立运行实测 `ModuleNotFoundError: No module named
  'cup_grasp_demo'`；debug.py 拉起它只靠父进程 PYTHONPATH 兜住。
- **修法**：`parents[1]` → `parents[2]`，删重复的第 12 行。

#### [ ] P2-22 `model_adapter` 未接线且自身有 bug（TODO.md 已记接线，此处记修）

- **位置**：`vision/inference/model_adapter.py:25-64`
- **问题**：接线前必须先修两处——① `int(best[0,0,0])` 返回的是第一个 anchor 的类别
  id，对多类模型毫无意义；② 从不校验 `det.shape[1]-4-nc == proto.shape[1] == 32`，
  通道不匹配会带错形状流到 `cup_perception.decode()` 才报错。
- **修法**：返回值改 per-detection 类别数组或删掉第三个返回值；补 proto 通道一致性
  断言。（与 TODO.md「model_adapter.py 接线」同批做。）

---

## 🟡 P3 —— 长尾排期

### 控制台输入健壮性

- [ ] **P3-1** `control_console.py:314-317`、`result_feedback.py:44`：`isdigit()+int()`
  被 Unicode 上标数字（`g²`）/超长数字击穿抛未捕获 ValueError。修：
  `isascii() and isdigit()` 或 try/except。
- [ ] **P3-2** `control_console.py:302-306`：非 UTF-8 输入 `UnicodeDecodeError` 未捕；
  子进程先死时主循环仍阻塞在 `input()`（reader 线程置 stop 叫不醒提示符）。
- [ ] **P3-3** reload 后菜单/快捷键不重建（TODO.md「UX 三小尾件」已记，实锤补充：
  被删动作键位仍显示、按下去 `rejected(unknown_action)`）。

### 配置卫生

- [ ] **P3-4** `configs/green_cup.json:88-98`：`shake_study` 零引用死键（真源是
  `shake.joint_motion_cost`）。
- [ ] **P3-5** 出厂绝对路径残留 3 处：`green_open_cup/home_table_scene.json:23` 的
  `source: /home/test2/...`、`configs/calibration/handeye_result.json:262` 与
  `configs/installation/camera.json:262` 的 `source_dataset: /home/test2/...`
  （与 P1-3 同批处理）。
- [ ] **P3-6** `configs/green_cup.json:131` 与 `cup_perception.py:24` 双处硬编码
  `/usr/lib/python3.14/dist-packages`（板上 3.12/3.14 混跑）；`detector.py:65-70`
  的 ORT 兜底还缺 `exc.name` 判断、把版本相关路径 append 进 sys.path。
- [ ] **P3-7** `configs/installation/camera.json` 是过期副本（sha256 与活动标定不一致），
  仅打包时被 `package_release.py:45` 覆盖——建议改为打包时直接复制活动标定。

### 打包与交付

- [ ] **P3-8** `package_release.py:12-16,37,48-50`：换机 `relative_to` 未捕获且失败
  不清理半成品目录（之后永远打不了包）；`green_open_cup/config.json` 不在重写列表，
  同包三种闸门状态不一致。
- [ ] **P3-9** `ensure_can_link.sh:6,12-18`：`DICE_CAN_INTERFACE` 全仓库无人导出
  （与 `cfg['channel']` 脱钩，改 can1 后脚本仍去 can0 报平安 exit 0）；只验 UP
  不验位速率。

### 视觉诊断工具

- [ ] **P3-10** `frame_io.py:181`：CLI 默认 `--grasp-config` 指向不存在的
  `vision/capture/CURRENT_GRASP.json`（真文件在 green_open_cup/）；`:303` 对从不
  落盘的 PNG 做 sha256（采集端只写 npz+json）——跑完最后一步才崩。
- [ ] **P3-11** `frame_io.py:129 vs 138`：空红布掩码在 RANSAC 之后才检查——
  `_plane` 抛 numpy 内部错误，第 138 行的 `Red workspace not visible` 永不可达。
- [ ] **P3-12** `circle_rim.py:348-350`：缺空轮廓保护（`max() arg is an empty
  sequence`），且该消息不在 `green_pipeline.py:447-449` 瞬态重试白名单。
- [ ] **P3-13** `green_rtsp.py:114-125`：写线程 `wait → 取槽 → clear` 顺序可吞唤醒
  （偶发 1s 停帧）；SIGKILL 可能残留 gst-launch 子进程。
- [ ] **P3-14** `realsense_session.py:161-193`：在途请求保护可被绕过（读线程开始服务
  即清 `_request`，第二次并发调用会通过检查）——当前 Workflow 单线程，潜伏。
- [ ] **P3-15** `vision/inference/yolo_seg.py`：三个解码器 + `YoloSegmentor` 死代码，
  测试只覆盖死路径（产线走 `cup_perception.decode`）；`decode_standard2` 掩码用
  `logits>0` 与产线 `sigmoid>threshold` 不等价。
- [ ] **P3-16** `vision/capture/config.py:38-44`：`calibration_file`/`calibration_digest()`
  无人调用，README:52 却宣称它做一致性校验——真正生效的是
  `configs/green_cup.json:7` 的 `calibration` 键。二选一：接线或删字段改 README。
- [ ] **P3-17** `planar_scene.py:15-20,44`：会话路径以调用者 cwd 为基准、采集子进程以
  ROOT 为基准，相对 `--session` 时分叉。

### 时钟/字段一致性

- [ ] **P3-18** 反馈年龄全用 wall clock（`joint_stream.py:81-89` 等）：NTP 跳变即
  误报"不新鲜"（硬件 epoch 戳所限；可改 `now_monotonic - (observed - stamp)` 折算）。
- [ ] **P3-19** `hold_error` 丢异常类型（`joint_execution.py:416` 用 `str()`，error 用
  `type+str`）；`feedback_freshness_limit_s` 只在配了 command_rate_hz 才记录。
- [ ] **P3-20** `shake_execution.py:373/390` + `shake_tracking.py:146`：
  `check_delivery` 的"早到"分支不可达（index 由同一 elapsed 反算）；lag 测的是
  选点到检查点的内部耗时，不是指令延迟。
- [ ] **P3-21** `green_runtime.py:34-54,189-191`：`_spawn` 失败泄漏日志 fd 且对象
  不 close（被 Startup future + atexit 持有到进程退出）。
- [ ] **P3-22** `fast_feedback.py:11`：缓存复用门限写死 0.1（有新鲜读回退兜底，
  仅性能项）。

---

## 📌 为什么 464 个测试全绿还漏了这些

两个直接实锤 + 一个系统性原因：

1. `test_sdk_restart.py` 的假 worker 不复刻真 worker 的 `.log`/`output.exists()` 语义，
   还伪造了 run/snapshot 不产出的 `tx` 块（→ P2-1 在真机永不出现的场景里绿了）。
2. `test_green_control.py` 的 `_pipe_session` 写完即关 writer，EOF 使 select 恒
   ready——stop 盲区测试通过的理由与实际语义无关（→ P2-3）。
3. TODO.md 已记录的"测试配置漂移 21 键"同类问题：测试在验证一套偏离交付的参数组合。

**修复时的通用要求**：每条修前存档锚点 → 修 → 新增测试做 stash 回归验证
（在旧代码上必失败才证明抓住 bug）→ 全量 464+ 通过 → 提交。

## 建议修复顺序

1. **P1-1 + P2-5/6/7/8 新鲜度剩余接线**（同病根一次收干净；P1-1 在现场踩过的失败链上）
2. **P1-2**（游戏程序马上要用 rounds 协议）
3. **P1-3 + P3-5**（绝对路径写读两侧一起堵）
4. **P2-1/2/3/4 SDK 自愈与连跑**（既然做了就要真的能用）
5. **P2-10/11/12 摇骰热路径与清理**
6. **P2-13..17 入口健壮性**（手误即崩类）
7. **P2-18..22 视觉采集**
8. P3 按主题批量

---
*产出：2026-09-24 全项目深挖（4 路并行审计 + P1 逐条人工复核）。基线 5d6f4c1。*
