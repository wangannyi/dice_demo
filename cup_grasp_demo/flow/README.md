# flow/ —— 绿杯主流程

| 分组 | 模块 |
| --- | --- |
| 阶段机 | green_pipeline（HOME→CAPTURE→PLAN→APPROACH→GRIP→LIFT→SHAKE→LOWER→OPEN→RETURN_HOME）、green_control（CONTROL 常驻 JSON 指令） |
| 运行时 | green_runtime（相机/模型/SDK 预热复用）、green_sdk_worker（CAN 执行子进程）、green_rtsp（摄像头推流）、green_prepared（路径复用） |
| 感知 | green_yolo、cup_perception、cup_recheck、cup_selection、green_stereo_rim、green_image_rim、green_cup_geometry、green_capture |
| 运动 | shake、shake_execution、joint_delivery、joint_profile、planar_*、batched_limits、contact_geometry、green_hand_execution、fast_feedback |
| 规划 | green_cup_planning、grasp、direct_grasp、grasp_cli、parameters、pipeline_home、tcp_overlay |
| 工具 | debug.py（多命令 CLI 入口）、run_debug.sh、core、hardware、session_storage |
| 数据 | green_open_cup/、index_joint_center/、config.json、fixtures/ |

静态动作（HOME 角度、摇晃配方、反馈手势）在 configs/actions/；动态抓取参数在 configs/green_cup.json。
