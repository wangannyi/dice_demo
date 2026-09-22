# flow/ —— 绿杯主流程

| 分组 | 模块 |
| --- | --- |
| 阶段机 | green_pipeline（HOME→CAPTURE→PLAN→APPROACH→GRIP→LIFT→SHAKE→LOWER→OPEN→RETURN_HOME）、green_control（常驻统一指令调度器：advance + action） |
| 运行时 | green_runtime（相机/模型/SDK 预热复用）、green_sdk_worker（CAN 执行子进程）、green_rtsp（摄像头推流）、green_prepared（路径复用） |
| 感知 | green_yolo、cup_perception、cup_selection、green_stereo_rim、green_image_rim、green_cup_geometry、green_capture |
| 运动 | shake、shake_execution、joint_delivery、joint_profile、joint_stream、shake_tracking、shake_readback、batched_limits、contact_geometry、grasp_execution、green_hand_execution、fast_feedback |
| 规划 | green_cup_planning、grasp、direct_grasp、parameters、feedback_execution、feedback_sequence |
| 工具 | debug.py（green-detect + pipeline 入口）、run_debug.sh、core、hardware、session_storage、planar_scene（桌面登记） |
| 数据 | green_open_cup/、joint_test_config.json、fixtures/ |

静态动作在 configs/actions/；动态抓取参数在 configs/green_cup.json。
