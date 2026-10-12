"use strict";
const $ = (id) => document.getElementById(id);
const tab = Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) =>
  b.toString(16).padStart(2, "0"),
).join("");
let state,
  csrf = "",
  mode = "first",
  stepIndex = 0,
  sending = false,
  pendingAction = null;
let roiImage = new Image(),
  roiLoaded = "",
  dragStart = null,
  lastResultHash = "",
  imageKey = "";
let pendingInspectionAdvance = null;
const labels = {
  inspect: "设备检查",
  home: "回 HOME",
  first: "首次示教",
  plan: "生成路线",
  auto: "自动采样",
  solve: "重新求解",
  register: "登记固定板",
  restore: "恢复相机位置",
  apply: "应用并登记桌面",
  table: "重试桌面登记",
};
const workflows = {
  first: ["inspect", "first", "window", "plan", "auto", "register", "apply"],
  auto: ["inspect", "plan", "auto", "register", "apply"],
  restore: ["inspect", "restore", "apply"],
};
const copy = {
  home: ["回 HOME", "机械臂回到 HOME，灵巧手归零。", "回 HOME", []],
  inspect: [
    "准备设备与工作区",
    "退出占用设备的程序，再检查连接。检查时机械臂不移动。",
    "连接并检查设备",
    [
      "已退出其他设备控制程序",
      "相机已固定，连接线不会被机械臂拉扯",
    ],
  ],
  first: [
    "手动示教并保存姿态",
    "手动调整姿态，停稳后保存。至少 12 组，建议 15–20 组，并改变位置与朝向。",
    "开始手动采样",
    [
      "末端标定板已固定，桌面板已遮挡",
      "已确认示教方式，整块末端板清晰可见",
    ],
  ],
  window: [
    "设定整板可见范围",
    "框选所有示教姿态中标定板的可见范围，然后保存。",
    "在下方保存范围",
    [],
  ],
  plan: [
    "生成自动采样路线",
    "根据示教记录生成路线。修改可见范围后需重新生成。",
    "生成路线",
    [],
  ],
  auto: [
    "自动采样",
    "沿示教路线采样，完成后回 HOME 并将灵巧手归零。开始时无需看见标定板。",
    "开始自动标定",
    [
      "末端板安装关系与示教时一致，桌面板已遮挡",
      "已检查 HOME 和采样路线；各采样点整板可见",
      "相机、基座和周围障碍物的位置与示教时一致",
    ],
  ],
  register: [
    "登记桌面固定板",
    "露出桌面固定板，保持相机与基座不动。",
    "登记固定板",
    ["桌面固定板已露出且不会移动", "相机与机械臂基座自本次标定后未移动"],
  ],
  restore: [
    "通过固定板恢复相机位置",
    "先回 HOME，再拍摄固定板。基座或固定板移动过，请选择“首次安装”。",
    "开始恢复",
    [
      "基座和桌面固定板自登记后没有移动",
      "固定板在画面中清楚可见，使用的是原相机",
      "HOME 路径已经现场检查，工作区无人",
    ],
  ],
  apply: [
    "应用结果并登记桌面",
    "拆除末端板，确认机械臂在 HOME，露出红布桌面。",
    "应用并登记桌面",
    [
      "末端标定板已拆除，机械臂已在 HOME",
      "红布桌面已露出，相机与基座保持不动",
      "已核对页面底部的应用配置路径",
    ],
  ],
  table: [
    "重新登记桌面",
    "调整桌面后重试，登记成功后才可使用。",
    "重试桌面登记",
    ["末端板已拆除，机械臂已在 HOME，红布桌面已露出"],
  ],
};
const titles = { first: "首次安装", auto: "重新标定", restore: "相机位置变了" };
const subtitles = {
  first: "按步骤完成示教、采样和标定。",
  auto: "沿已保存的路线重新标定。",
  restore: "通过桌面固定板恢复相机位置。",
};
function text(id, value) {
  if ($(id).textContent !== String(value)) $(id).textContent = value;
}
function showError(message) {
  text("error", message);
  $("error").hidden = !message;
}
function badge(id, label, kind = "") {
  text(id, label);
  $(id).className = "badge " + kind;
}
function step() {
  return workflows[mode][stepIndex];
}
function prepared() {
  return [...$("preparation").querySelectorAll("input")].every(
    (input) => input.checked,
  );
}
function node(tag, content, className) {
  const el = document.createElement(tag);
  if (content != null) el.textContent = content;
  if (className) el.className = className;
  return el;
}
async function post(path, body = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-CSRF-Token": csrf,
      "X-Control-Tab": tab,
    },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "操作失败");
  return data;
}
async function refresh() {
  try {
    const response = await fetch("/api/status", {
      headers: { "X-Control-Tab": tab },
    });
    if (!response.ok) throw new Error("无法读取设备状态");
    state = await response.json();
    csrf = state.csrf;
    render();
    if (pendingInspectionAdvance && !state.active) {
      const pending = pendingInspectionAdvance;
      pendingInspectionAdvance = null;
      if (state.job?.id === pending.job && state.job.status === "completed" &&
          state.job.stop?.confirmed && state.inspection_ready && mode === pending.mode && step() === "inspect")
        selectStep(Math.min(stepIndex + 1, workflows[mode].length - 1));
    }
  } catch (error) {
    text("connection", "服务连接中断");
    badge("frame-status", "画面已过期", "warn");
    showError(
      "无法连接服务。正在重连；已有作业会按心跳超时请求停止，不会自动续跑。",
    );
  }
}
function selectStep(index) {
  stepIndex = index;
  const action = step(),
    c = copy[action];
  text("title", titles[mode]);
  text("subtitle", subtitles[mode]);
  $("steps").replaceChildren(
    ...workflows[mode].map((value, i) => {
      const li = node("li");
      const b = node("button");
      b.type = "button";
      b.append(
        node("span", String(i + 1).padStart(2, "0"), "step-count"),
        document.createTextNode(
          value === "window" ? "可见范围" : labels[value],
        ),
      );
      b.append(node("span", "", "step-status"));
      if (i === index) b.setAttribute("aria-current", "step");
      b.addEventListener("click", () => selectStep(i));
      li.append(b);
      return li;
    }),
  );
  text("step-title", c[0]);
  text("step-description", c[1]);
  text("step-number", "步骤 " + String(index + 1).padStart(2, "0"));
  text("start-action", c[2]);
  $("preparation").replaceChildren(
    ...c[3].map((item) => {
      const label = node("label", null, "check"),
        input = node("input");
      input.type = "checkbox";
      input.addEventListener("change", renderControls);
      label.append(input, node("span", item));
      return label;
    }),
  );
  $("window-panel").hidden = action !== "window";
  $("next-step").hidden = index === workflows[mode].length - 1;
  if (action === "window") loadRoi();
  if (state) renderControls();
}
// Completion is scoped to the latest attempt and its current input artifacts.
// A failed retry or newer upstream operation must never inherit an old success.
function completionFor(action) {
  if (!state || state.active || state.recovery_required) return null;
  if (["auto", "register", "apply"].includes(action) && state.state?.home_required) return null;
  if (action === "inspect") return state.inspection_ready && !state.busy.length
    ? { title: "设备检查已完成", message: "设备已就绪，可进入下一步。" } : null;
  if (action === "window") return state.state?.teaching_dataset && state.actions.plan?.ready
    ? { title: "可见范围已保存", message: "请继续生成自动采样路线。" } : null;
  const attempts = state.jobs || [];
  const job = attempts.find((j) => j.action === action || (action === "apply" && j.action === "table"));
  if (!job || job.status !== "completed" || job.needs_device_check) return null;
  const upstream = {
    first: [], plan: ["first"], auto: ["first", "plan"],
    register: ["first", "auto", "solve", "restore"], restore: ["register"],
    apply: ["first", "auto", "solve", "restore", "register"],
  }[action] || [];
  if (attempts.some((j) => upstream.includes(j.action) && j.started > job.started)) return null;
  const artifacts = {first: state.state?.teaching_dataset, plan: state.state?.plan,
    auto: state.result, register: state.state?.registration, restore: state.result,
    apply: state.applied && state.table_ready};
  if (!artifacts[action]) return null;
  const messages = {
    first: ["示教与求解已完成", "采样数据已保存。请继续绘制可见范围。"],
    plan: ["采样路线已生成", "请进入自动采样。"],
    auto: ["自动采样与求解已完成", "结果已保存，尚未应用。"],
    register: ["固定板登记已完成", "请继续应用结果并登记桌面。"],
    restore: ["相机位置恢复已完成", "恢复结果已保存，尚未自动应用。"],
    apply: ["本次标定流程已完成", "标定结果已应用，桌面登记已完成。"],
  };
  return messages[action] ? {title: messages[action][0], message: messages[action][1]} : null;
}
function renderCompletion(action) {
  const done = completionFor(action);
  const next = workflows[mode][stepIndex + 1];
  $("completion-notice").hidden = !done;
  if (done) {
    text("completion-title", "✓ " + done.title);
    let message = done.message;
    if (action === "auto" && state.state?.home_return?.confirmed) {
      message += " 已回 HOME 并确认静止。";
      if (state.state?.hand_home?.command_sent) message += " 灵巧手归零指令已发送。";
    }
    if (["first", "auto", "restore"].includes(action) && state.result) {
      const r = state.result;
      message += r.quality_passed ? " 误差达标。" : " 误差未达标，临时使用需管理员批准。";
    }
    text("completion-message", message);
  }
  $("completion-result-link").hidden = !done || !["first", "auto", "restore"].includes(action);
  $("start-action").className = done ? "secondary" : "primary";
  $("next-step").className = done ? "primary" : "secondary";
  text("next-step", done && next ? `下一步：${next === "window" ? "绘制可见范围" : labels[next]} →` : "下一步 →");
  const buttons = $("steps").querySelectorAll("button");
  buttons.forEach((button, i) => {
    const completed = !!completionFor(workflows[mode][i]);
    button.classList.toggle("step-done", completed);
    button.classList.toggle("step-next", !!done && i === stepIndex + 1);
    button.querySelector(".step-status").textContent = completed ? " · 已完成" : done && i === stepIndex + 1 ? " · 下一步" : "";
  });
  return done;
}
function renderControls() {
  if (!state) return;
  const action = step(),
    job = state.job,
    active = state.active,
    own = state.owns_job;
  const readiness = state.actions[action];
  const completed = renderCompletion(action);
  if (completed) badge("job-badge", "本次操作已完成", "good");
  text("step-title", completed ? completed.title : copy[action][0]);
  text("step-description", completed ? "本步骤已完成。" : copy[action][1]);
  $("quality-policy").hidden = true;
  const hardware = !["plan", "window", "solve"].includes(action);
  const inspectionReady = action === "inspect" && state.inspection_ready && !active && !state.busy.length;
  let reason = "";
  if (state.readonly) reason = "当前为只读界面预览，不执行标定。";
  else if (pendingInspectionAdvance && active)
    reason = "正在完成检查，请稍候。";
  else if (action === "inspect" && state.inspection_connected)
    reason = own
      ? "设备已连接，请点击“完成检查并进入下一步”。"
      : "设备已连接，请在原控制页面继续。";
  else if (active)
    reason = own
      ? "正在操作，请保持页面连接。"
      : "其他页面正在控制标定。此页面仅查看状态。";
  else if (hardware && state.busy.length)
    reason = "设备被其他程序占用，请先正常退出原程序，再点击“重新检测”。";
  else if (state.recovery_required && hardware && action !== "inspect")
    reason = "上次设备状态未确认，请先完成设备检查。";
  else if (state.state?.home_required && ["auto", "register", "apply"].includes(action))
    reason = state.state?.hand_home_required
      ? "灵巧手归零指令未完成，请重试回 HOME。"
      : state.state?.home_check?.at_home === false
      ? "尚未到 HOME，请回 HOME 后继续。"
      : "HOME 状态待确认，请重新检查设备。";
  else if (readiness && !readiness.ready) reason = readiness.reason;
  else if (inspectionReady) reason = "设备已就绪，请进入下一步。";
  else if (completed) reason = "";
  else if (!prepared()) reason = (state.state?.home_check?.at_home ? "已确认在 HOME。" : "") + "请勾选准备事项。";
  else if (action === "window")
    reason = "请框选并保存可见范围。";
  else if (readiness?.source?.approval_required)
    reason = "当前结果需要管理员批准。";
  else if (readiness?.source?.approved && !readiness.source.quality_passed)
    reason = "管理员已批准临时使用，可继续。";
  text("prerequisite", reason);
  $("prerequisite").classList.toggle("ready-notice", !!inspectionReady || (action === "inspect" && state.inspection_connected));
  text("start-action", completed ? ({inspect: "重新检查设备", first: "重新示教", plan: "重新生成路线", auto: "重新自动标定", register: "重新登记固定板", restore: "重新恢复位置", apply: "重新应用并登记桌面"}[action] || copy[action]?.[2]) : copy[action]?.[2]);
  $("start-action").hidden = active || action === "window";
  $("start-action").disabled =
    sending ||
    state.readonly ||
    active ||
    !prepared() ||
    (hardware && state.busy.length > 0) ||
    (readiness && !readiness.ready) ||
    (state.recovery_required && hardware && action !== "inspect");
  const sampling =
    active &&
    own &&
    job?.action === "first" &&
    job?.stage?.script === "calibration/calibrate.py";
  $("capture").hidden = !sampling;
  $("capture").disabled = sending || job?.status === "stopping";
  $("finish").hidden = !(
    active &&
    own &&
    (sampling || job?.action === "inspect")
  );
  text("finish", job?.action === "inspect"
    ? pendingInspectionAdvance ? "正在完成检查…" : state.inspection_connected ? "完成检查并进入下一步 →" : "结束设备检查"
    : "结束采样并求解");
  $("finish").disabled = sending || job?.status === "stopping" || !!pendingInspectionAdvance;
  $("cancel").hidden = !(active && own);
  $("cancel").disabled = sending || job?.status === "stopping";
  $("inspect-start").disabled =
    state.readonly || active || state.busy.length > 0;
  $("save-window").disabled =
    state.readonly || active || !roiImage.naturalWidth;
  $("return-home").disabled = sending || state.readonly || active || state.busy.length > 0 || state.recovery_required;
  $("return-home").className = state.state?.home_required ? "primary" : "secondary";
  $("next-step").disabled = active || (state.state?.home_required && ["auto", "register", "apply"].includes(action)) || (action === "inspect" && (!state.inspection_ready || state.busy.length > 0));

}
function render() {
  text(
    "connection",
    state.readonly
      ? "只读预览"
      : state.active
        ? state.job?.action === "inspect"
          ? state.inspection_connected ? "设备已连接 · 实时预览" : "正在检查设备连接"
          : "标定作业进行中"
        : "服务已连接",
  );
  text("target-path", state.pipeline_config);
  text(
    "engineer-open",
    state.engineer_unlocked ? "工程师已解锁 · 管理 ↗" : "工程师入口 ↗",
  );
  let notice = "";
  if (state.readonly) notice = "只读预览模式：所有硬件操作已禁用。";
  else if (state.busy.length)
    notice =
      "设备占用中 · " +
      "请退出占用程序后重新检测。";
  else if (state.recovery_required)
    notice =
      "请先完成设备检查，再继续标定。";
  text("global-notice", notice);
  $("global-notice").hidden = !notice;
  const job = state.job,
    progress = job?.progress,
    detection = job?.detection,
    frame = job?.frame;
  const names = {
    running: "正在进行",
    stopping: "正在请求停止",
    completed: "作业完成",
    cancelled: "作业已取消",
    failed: "作业失败",
    interrupted: "服务曾中断",
  };
  badge(
    "job-badge",
    job?.needs_device_check
      ? "设备待检查"
      : job?.action === step()
        ? names[job.status] || job.status
        : state.active
          ? "其他步骤进行中"
          : "待开始",
    ["failed", "interrupted"].includes(job?.status)
      ? "bad"
      : job?.status === "completed"
        ? "good"
        : "",
  );
  if (step() === "inspect" && state.inspection_ready && !state.active && !state.busy.length)
    badge("job-badge", "准备已就绪", "good");
  if (step() === "inspect" && state.inspection_connected)
    badge("job-badge", "设备连接已完成", "good");
  if (job?.needs_device_check)
    text(
      "stop-status",
      "未确认停止，请现场检查，必要时使用硬件急停。",
    );
  else if (job?.error)
    text(
      "stop-status",
      operatorFailure(job.error) +
        (job.stop?.confirmed
          ? " · 机械臂已停止。"
          : " · 未确认机械臂停止，请现场检查。"),
    );
  else if (job?.status === "stopping")
    text("stop-status", "正在停止，请等待确认。");
  else if (job?.status === "completed" && ["home", "auto"].includes(job.action) &&
      state.state?.home_return?.confirmed && state.state?.hand_home?.command_sent)
    text("stop-status", "机械臂已回 HOME，灵巧手归零指令已发送。");
  else
    text(
      "stop-status",
      `关闭页面或断线超过 ${state.heartbeat_timeout_s} 秒会请求停止；重连后不会自动续跑。`,
    );
  const age = job?.frame_age_s ?? Infinity;
  if (frame && job) {
    $("camera-empty").hidden = true;
    $("live-image").hidden = false;
    $("image-footer").hidden = false;
    const key = job.id + ":" + frame.time;
    if (key !== imageKey) {
      imageKey = key;
      $("live-image").src =
        `/api/image?kind=preview&job=${encodeURIComponent(job.id)}&t=${frame.time}`;
    }
    text("image-size", `${frame.width} × ${frame.height}`);
    badge(
      "frame-status",
      !state.active ? "最后一帧" : age > 3 ? "画面已过期" : "实时画面",
      state.active && age <= 3 ? "good" : "warn",
    );
  } else {
    $("live-image").hidden = true;
    $("image-footer").hidden = true;
    $("camera-empty").hidden = false;
    badge("frame-status", state.active ? "等待相机画面" : "未连接相机");
  }
  const freshDetection =
    detection && Date.now() / 1000 - detection.time < 5 && state.active;
  text(
    "detection-label",
    freshDetection
      ? detection.detected
        ? "已识别"
        : "未识别"
      : progress?.quality
        ? "上次采样已识别"
        : "等待检测",
  );
  const quality = freshDetection ? detection.quality : progress?.quality;
  text(
    "reprojection",
    quality?.reprojection_rms_px != null
      ? quality.reprojection_rms_px.toFixed(3) + " px"
      : "—",
  );
  text(
    "sample-count",
    progress?.accepted != null ? `${progress.accepted} 组` : "—",
  );
  text(
    "detection-reason",
    freshDetection && !detection.detected
      ? "标定板未识别，请检查是否清晰可见。"
      : "采手眼数据时，末端板必须固定牢固，并遮挡桌面固定板。",
  );
  $("progress-area").hidden = !state.active || job?.action === "inspect";
  const display = jobProgress(job);
  $("progress-area").querySelector("label").hidden = !display.counted;
  $("sample-progress").hidden = !display.counted;
  $("sample-progress").max = display.total || 1;
  $("sample-progress").value = display.count;
  text("progress-label", display.counted ? `${display.count} / ${display.total}` : "");
  text("progress-message", display.message);
  renderControls();
  renderResult();
  window.calibrationGallery?.update(state);
  renderLiveRange();
  renderHistory();
}
function operatorFailure(error) {
  const value = String(error || "");
  if (/no fresh .*Revo2 position feedback|missing Revo2 position timestamp/i.test(value))
    return "未收到手指位置反馈，请检查灵巧手通信。";
  if (/Revo2|灵巧手|手指/i.test(value)) return "灵巧手复位未完成，请检查连接和活动空间后重试。";
  if (/ChArUco|corners|markers|标定板/i.test(value)) return "标定板识别失败，请调整画面后重试。";
  if (/camera|RealSense|相机/i.test(value)) return "相机连接异常，请检查后重试。";
  return "操作未完成，请查看“操作记录与诊断”。";
}
function jobProgress(job) {
  const progress = job?.progress || {}, action = job?.action;
  const script = job?.stage?.script;
  const home = action === "home" || script === "web_home_return" ||
    ["HOME", "HAND_HOME", "HAND_COMMAND_SENT", "RETURN_HOME", "HOME_READY"].includes(progress.phase);
  const table = action === "apply" || action === "table";
  const total = progress.total;
  // Only automatic hand-eye sampling reports a known number of target poses.
  const counted = action === "auto" && !home &&
    script === "calibration/auto_collect.py" && Number.isInteger(total) && total > 0;
  const count = Number.isInteger(progress.accepted) && progress.accepted >= 0 ? progress.accepted : 0;
  const tableStages = {
    "calibration/apply_result.py": "正在应用标定结果…",
    "calibration/tools/table_capture.py": "正在拍摄桌面并拟合桌面平面…",
    "calibration/tools/register_home_table.py": "正在保存桌面配置…",
  };
  const message = table
    ? tableStages[script] || (action === "apply" ? "正在备份配置并准备应用结果…" : "正在准备桌面拍摄…")
    : { HOME: "正在回 HOME…", RETURN_HOME: "正在回 HOME…", HAND_HOME: "正在将灵巧手归零…",
        HAND_COMMAND_SENT: "灵巧手归零指令已发送。", HOME_READY: "已回 HOME，灵巧手归零指令已发送。", READY: "正在准备采样…", ACCEPTED: "样本已保存，继续采样…" }[progress.phase] ||
      (action === "first" && ["sampling", "capturing"].includes(progress.phase)
        ? progress.last_accepted === false ? "样本未保存，请调整姿态后重试。"
          : progress.phase === "capturing" ? "正在采样，请保持静止…" : `已保存 ${count} 组手眼样本`
        : /^MOVING to sample \d+$/.test(progress.phase || "") ? "正在前往下一采样点…"
          : script === "calibration/calibrate.py" || script === "calibration/tools/reprocess_dimensions.py" ? "正在计算标定结果…"
            : "正在处理，请稍候…");
  return {counted, total: counted ? total : null, count, message};
}
function liveRangeGeometry(v, frame, imageWidth, imageHeight) {
  if (!v?.available || !v.live_compatible || !frame ||
      frame.width !== v.width || frame.height !== v.height ||
      imageWidth !== frame.width || imageHeight !== frame.height) return null;
  const [x0, y0, x1, y1] = v.window_xyxy;
  return {x: x0, y: y0, width: x1-x0, height: y1-y0, viewBox: `0 0 ${v.width} ${v.height}`};
}
function renderLiveRange() {
  if (!state) return;
  const v = state.visibility, frame = state.job?.frame, img = $("live-image");
  const geometry = img.complete && !img.hidden
    ? liveRangeGeometry(v, frame, img.naturalWidth, img.naturalHeight) : null;
  $("live-range").toggleAttribute("hidden", !geometry);
  if (geometry) {
    $("live-range").setAttribute("viewBox", geometry.viewBox);
    for (const key of ["x", "y", "width", "height"])
      $("live-range-box").setAttribute(key, geometry[key]);
    text("live-range-caption", `绿色框：标定可见范围${!state.active ? "（最后一帧）" : ""}`);
  } else {
    text("live-range-caption", !v?.available ? "尚未保存标定可见范围。" : !frame
      ? "连接相机后，将在视频中显示绿色可见范围框。"
      : !v.live_compatible ? "画面配置尚未匹配已保存范围，未叠加边框；请重新连接检查相机。"
      : "正在等待匹配的相机画面…");
  }
}
$("live-image").addEventListener("load", renderLiveRange);
$("live-image").addEventListener("error", () => {
  $("live-range").setAttribute("hidden", "");
  text("live-range-caption", "画面读取失败，可见范围框已隐藏。");
});
function renderResult() {
  const r = state.result;
  $("result-empty").hidden = !!r;
  $("result-content").hidden = !r;
  if (!r) {
    badge("result-badge", "尚无结果");
    return;
  }
  text("result-context", state.active && ["first", "auto", "restore", "solve"].includes(state.job?.action)
    ? "正在标定，下方为上次结果。" : "当前已保存的标定结果");
  const positionLimit = r.thresholds?.position_m ?? 0.005,
    angleLimit = r.thresholds?.angle_deg ?? 2;
  text("position-threshold", `门槛 ≤ ${positionLimit * 1000} mm`);
  text("angle-threshold", `门槛 ≤ ${angleLimit}°`);
  badge(
    "result-badge",
    r.quality_passed ? "质量达标" : "质量未达标",
    r.quality_passed ? "good" : "warn",
  );
  text(
    "position-error",
    r.max_position_mm == null ? "—" : r.max_position_mm.toFixed(2) + " mm",
  );
  text(
    "angle-error",
    r.max_angle_deg == null ? "—" : r.max_angle_deg.toFixed(2) + "°",
  );
  text("result-samples", r.sample_count + " 组");
  $("retry-table").hidden = !state.applied || state.table_ready;
  $("retry-table").disabled = state.active || state.readonly;
  $("solve-again").disabled =
    state.active || state.readonly || !state.actions.solve.ready;
  text("application-status", state.applied ? "已安装结果" : "尚未应用");
  text(
    "table-status",
    state.table_ready
      ? "桌面登记已完成"
      : state.applied
        ? "桌面待登记，运行仍受限"
        : "等待应用后登记桌面",
  );
  text(
    "result-message",
    r.data.extrinsic_method === "fixed_reference_restore"
      ? "相机位置已恢复。误差来自原标定，本次未独立验证精度。"
      : r.quality_passed
        ? "误差达标，请继续后续步骤。"
        : "误差未达标，建议重新标定；临时使用需管理员批准。",
  );
  const resultKey = `${r.sha256}:${positionLimit}:${angleLimit}`;
  if (lastResultHash !== resultKey) {
    lastResultHash = resultKey;
    const table = node("table"),
      head = node("tr");
    ["样本组", "位置误差 / mm", "角度误差 / °", "验证方式", "门槛判定"].forEach((v) =>
      head.append(node("th", v)),
    );
    const thead = node("thead");
    thead.append(head);
    table.append(thead);
    table.setAttribute("aria-label", "每组标定误差");
    thead.querySelectorAll("th").forEach((th) => th.scope = "col");
    const tbody = node("tbody");
    (r.data.all_sample_errors_m_deg || []).forEach((value, i) => {
      const row = node("tr");
      const vals = [
        String(i + 1),
        (value[0] * 1000).toFixed(2),
        value[1].toFixed(2),
        "全样本拟合",
        value[0] <= positionLimit && value[1] <= angleLimit ? "通过" : "未通过",
      ];
      vals.forEach((v, j) =>
        row.append(
          node(
            "td",
            v,
            (j === 1 && value[0] > positionLimit) || (j === 2 && value[1] > angleLimit)
              ? "fail-cell"
              : "",
          ),
        ),
      );
      tbody.append(row);
    });
    (r.data.holdout_errors_m_deg || []).forEach((value, i) => {
      const row = node("tr");
      [
        (r.data.holdout_indices || [])[i] == null ? "未记录" : String(r.data.holdout_indices[i] + 1),
        (value[0] * 1000).toFixed(2),
        value[1].toFixed(2),
        "留出验证",
        value[0] <= positionLimit && value[1] <= angleLimit ? "通过" : "未通过",
      ].forEach((v, j) =>
        row.append(
          node(
            "td",
            v,
            (j === 1 && value[0] > positionLimit) || (j === 2 && value[1] > angleLimit)
              ? "fail-cell"
              : "",
          ),
        ),
      );
      tbody.append(row);
    });
    table.append(tbody);
    $("residuals").replaceChildren(table);
    text(
      "matrix",
      "相机坐标 → 机械臂基座坐标\n" +
        JSON.stringify(r.data.T_base_camera, null, 2),
    );
    text(
      "result-source",
      "当前结果：" + r.path + " · SHA256 " + r.sha256.slice(0, 16),
    );
  }
}
function renderHistory() {
  const table = node("table"),
    header = node("tr");
  ["时间", "操作", "状态"].forEach((v) => header.append(node("th", v)));
  const thead = node("thead");
  thead.append(header);
  table.append(thead);
  const tbody = node("tbody");
  for (const job of state.jobs.slice(0, 10)) {
    const row = node("tr");
    [
      new Date(job.started * 1000).toLocaleString("zh-CN"),
      labels[job.action] || job.action,
      {
        completed: "完成",
        failed: "失败",
        cancelled: "已取消",
        running: "进行中",
        stopping: "停止中",
        interrupted: "中断",
      }[job.status] || job.status,
    ].forEach((v) => row.append(node("td", v)));
    tbody.append(row);
  }
  table.append(tbody);
  $("history").replaceChildren(table);
  text("job-log", [state.job?.error, state.job?.detection?.reason,
    state.job?.progress && JSON.stringify(state.job.progress, null, 2), state.job?.log,
    state.busy.length ? JSON.stringify(state.busy, null, 2) : null].filter(Boolean).join("\n\n") || "尚无作业日志。");
}
async function prepareAction(action) {
  showError("");
  await refresh();
  if (state.active) throw new Error("请先结束当前作业。");
  if (!["solve", "home"].includes(action) && !prepared())
    throw new Error("请先逐项确认现场准备事项。");
  pendingAction = action;
  const source = state.actions[action]?.source;
  if (source?.approval_required && !state.engineer_unlocked) {
    openEngineer();
    return;
  }
  const c = copy[action] || [
    labels[action],
    "使用当前数据重新计算，不会移动机械臂。",
  ];
  text("confirm-title", c[0]);
  text("confirm-description", c[1]);
  $("motion-confirm").hidden = !["auto", "restore", "home"].includes(action);
  $("motion-checked").checked = false;
  $("quality-confirm").hidden = !source?.approval_required;
  $("quality-checked").checked = false;
  $("reason").value = "";
  $("confirm-error").hidden = true;
  const summary = source?.summary;
  text(
    "quality-detail",
    summary
      ? `误差未达标：位置 ${summary.max_position_mm?.toFixed(2) ?? "未知"} mm，最大角度误差 ${summary.max_angle_deg?.toFixed(2) ?? "未知"}°。该结果门槛为 ${(summary.thresholds?.position_m ?? 0.005) * 1000} mm / ${summary.thresholds?.angle_deg ?? 2}°。`
      : "原标定未达标，本次恢复后仍需管理员批准。",
  );
  $("confirm-dialog").showModal();
}
async function perform(action, body) {
  sending = true;
  renderControls();
  try {
    await post("/api/start", { action, prepared: true, ...body });
    $("confirm-dialog").close();
    showError("");
    await refresh();
  } finally {
    sending = false;
    renderControls();
  }
}
function openEngineer() {
  text(
    "pin-help",
    state.pin_configured
      ? "PIN 由 K3 本地管理员设置；不会显示在页面或日志中。"
      : "尚未配置 PIN。请在 K3 本地运行 bash calibrate_web.sh --set-pin。",
  );
  $("pin-error").hidden = true;
  $("pin").value = "";
  $("engineer-dialog").showModal();
}
function guarded(fn) {
  return async (...args) => {
    try {
      await fn(...args);
    } catch (error) {
      showError(error.message);
    }
  };
}
$("confirm-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const action = pendingAction;
  try {
    const motion = ["auto", "restore", "home"].includes(action),
      source = state.actions[action]?.source;
    if (motion && !$("motion-checked").checked)
      throw new Error("请确认机械臂运动条件。");
    if (
      source?.approval_required &&
      (!$("quality-checked").checked || $("reason").value.trim().length < 5)
    )
      throw new Error("请确认误差并填写至少 5 字的临时使用原因。");
    $("confirm-submit").disabled = true;
    await perform(action, {
      motion_confirmed: $("motion-checked").checked,
      quality_confirmed: $("quality-checked").checked,
      approved_sha256: source?.sha256,
      reason: $("reason").value.trim(),
    });
  } catch (error) {
    text("confirm-error", error.message);
    $("confirm-error").hidden = false;
  } finally {
    $("confirm-submit").disabled = false;
  }
});
$("confirm-back").addEventListener("click", () => $("confirm-dialog").close());
$("engineer-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await post("/api/unlock", { pin: $("pin").value });
    $("pin").value = "";
    $("engineer-dialog").close();
    await refresh();
    if (pendingAction) await prepareAction(pendingAction);
  } catch (error) {
    text("pin-error", error.message);
    $("pin-error").hidden = false;
  }
});
$("engineer-open").addEventListener("click", () => {
  pendingAction = null;
  openEngineer();
});
$("engineer-close").addEventListener("click", () =>
  $("engineer-dialog").close(),
);
$("engineer-lock").addEventListener(
  "click",
  guarded(async () => {
    await post("/api/lock");
    $("engineer-dialog").close();
    await refresh();
  }),
);
$("modes").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-mode]");
  if (!button) return;
  mode = button.dataset.mode;
  document
    .querySelectorAll("[data-mode]")
    .forEach((b) => b.removeAttribute("aria-current"));
  button.setAttribute("aria-current", "page");
  selectStep(0);
});
$("start-action").addEventListener(
  "click",
  guarded(() => prepareAction(step())),
);
$("inspect-start").addEventListener(
  "click",
  guarded(async () => {
    if (step() !== "inspect") selectStep(0);
    if (!prepared()) throw new Error("请在右侧确认准备事项，再连接设备。");
    await prepareAction("inspect");
  }),
);
$("return-home").addEventListener("click", guarded(() => prepareAction("home")));
$("solve-again").addEventListener(
  "click",
  guarded(() => prepareAction("solve")),
);
$("retry-table").addEventListener(
  "click",
  guarded(async () => {
    if (step() !== "apply") selectStep(workflows[mode].length - 1);
    if (!prepared())
      throw new Error(
        "请先确认右侧 HOME、末端板拆除及桌面准备事项，再点击重试。",
      );
    await prepareAction("table");
  }),
);
$("refresh").addEventListener(
  "click",
  guarded(async () => {
    showError("");
    await refresh();
  }),
);
$("next-step").addEventListener("click", () =>
  selectStep(Math.min(stepIndex + 1, workflows[mode].length - 1)),
);
$("capture").addEventListener(
  "click",
  guarded(async () => {
    await post("/api/command", { command: "sample" });
    showError("");
  }),
);
$("finish").addEventListener(
  "click",
  guarded(async () => {
    const advance = state.job?.action === "inspect" && state.inspection_connected;
    if (advance) pendingInspectionAdvance = {job: state.job.id, mode};
    try {
      await post("/api/command", { command: "finish" });
    } catch (error) {
      pendingInspectionAdvance = null;
      throw error;
    }
    renderControls();
    showError("");
  }),
);
$("cancel").addEventListener(
  "click",
  guarded(async () => {
    pendingInspectionAdvance = null;
    await post("/api/cancel");
    await refresh();
  }),
);
const roiFields = ["roi-x0", "roi-y0", "roi-x1", "roi-y1"];
function roi() {
  return roiFields.map((id) => Number($(id).value));
}
function drawRoi() {
  const canvas = $("window-canvas"),
    context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (roiImage.naturalWidth) context.drawImage(roiImage, 0, 0);
  const [x0, y0, x1, y1] = roi();
  context.strokeStyle = "#12a77e";
  context.lineWidth = 3;
  context.strokeRect(x0, y0, x1 - x0, y1 - y0);
  context.fillStyle = "#087c6722";
  context.fillRect(x0, y0, x1 - x0, y1 - y0);
}
function loadRoi() {
  const key = state?.state?.teaching_dataset || "reference";
  if (key === roiLoaded) return;
  roiImage = new Image();
  roiImage.onload = () => {
    roiLoaded = key;
    $("window-canvas").width = roiImage.naturalWidth;
    $("window-canvas").height = roiImage.naturalHeight;
    [0, 0, roiImage.naturalWidth, roiImage.naturalHeight].forEach(
      (v, i) => ($(roiFields[i]).value = v),
    );
    drawRoi();
    renderControls();
  };
  roiImage.onerror = () =>
    text("window-message", "尚无首张采样图，请先完成首次示教。");
  roiImage.src = "/api/image?kind=teaching&t=" + Date.now();
}
function point(event) {
  const canvas = $("window-canvas"),
    bounds = canvas.getBoundingClientRect();
  return [
    Math.round(
      Math.max(
        0,
        Math.min(
          canvas.width,
          ((event.clientX - bounds.left) / bounds.width) * canvas.width,
        ),
      ),
    ),
    Math.round(
      Math.max(
        0,
        Math.min(
          canvas.height,
          ((event.clientY - bounds.top) / bounds.height) * canvas.height,
        ),
      ),
    ),
  ];
}
$("window-canvas").addEventListener("pointerdown", (event) => {
  dragStart = point(event);
  $("window-canvas").setPointerCapture(event.pointerId);
});
$("window-canvas").addEventListener("pointermove", (event) => {
  if (!dragStart) return;
  const p = point(event);
  [
    Math.min(p[0], dragStart[0]),
    Math.min(p[1], dragStart[1]),
    Math.max(p[0], dragStart[0]),
    Math.max(p[1], dragStart[1]),
  ].forEach((v, i) => ($(roiFields[i]).value = v));
  drawRoi();
});
$("window-canvas").addEventListener("pointerup", () => (dragStart = null));
$("window-canvas").addEventListener("pointercancel", () => (dragStart = null));
roiFields.forEach((id) => $(id).addEventListener("input", drawRoi));
$("save-window").addEventListener(
  "click",
  guarded(async () => {
    await post("/api/window", { window_xyxy: roi() });
    text("window-message", "可见范围已保存。请重新生成路线。");
    showError("");
    await refresh();
  }),
);
window.addEventListener("pagehide", () => {
  if (state?.active && state.owns_job)
    fetch("/api/cancel", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRF-Token": csrf,
        "X-Control-Tab": tab,
      },
      body: "{}",
      keepalive: true,
    }).catch(() => {});
});
async function poll() {
  await refresh();
  if (state?.active && state.owns_job) {
    try {
      await post("/api/heartbeat");
    } catch (error) {
      showError(error.message);
    }
  }
  setTimeout(poll, 1000);
}
selectStep(0);
poll();
