/* Read-only gallery; never starts a job or requests device control. */
(() => {
  const el = (id) => document.getElementById(`gallery-${id}`);
  let catalog = { samples: [] }, refreshKey = null, listRequest = 0, frameRequest = 0;
  const selected = () => catalog.samples.find((s) => s.id === el("sample").value);
  function options(select, items, previous) {
    select.replaceChildren(...items.map(([value, label]) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      return option;
    }));
    if (items.some(([value]) => value === previous)) select.value = previous;
  }
  function link(id, url, filename) {
    el(id).hidden = !url;
    if (url) { el(id).href = url; el(id).download = filename; }
    else el(id).removeAttribute("href");
  }
  async function showFrame() {
    const serial = ++frameRequest;
    const sample = selected();
    const frame = sample?.frames.find((f) => f.id === el("frame").value);
    if (!frame) return;
    const kind = el("kind").value;
    const url = frame[kind];
    const label = `第 ${sample.number} 组 · ${frame.label} · ${kind === "raw" ? "原图" : "检测标注图"}`;
    const image = el("image");
    image.hidden = true;
    image.removeAttribute("src");
    image.alt = `${label}（历史采样图片）`;
    el("image-status").textContent = url ? "正在加载图片…" : "此帧图片未保存或文件已缺失。";
    image.onload = () => {
      if (serial !== frameRequest) return;
      image.hidden = false;
      el("image-status").textContent = `${label} · ${image.naturalWidth} × ${image.naturalHeight} px`;
    };
    image.onerror = () => {
      if (serial === frameRequest) el("image-status").textContent = "图片读取失败，请刷新图片列表后重试。";
    };
    if (url) image.src = url;
    link("download-image", url, `${sample.id}_${frame.id}_${kind}.png`);
    link("download-data", frame.data, `${sample.id}_${frame.id}.json`);
    el("frame-note").textContent = sample.frames.length > 1
      ? "可切换本组全部检测帧；“被选中”表示最终用于该组样本的图像。"
      : "此组仅有最终样本图片，未保存或未找到另外两帧的记录。";
    el("summary").textContent = "正在读取此帧数据…";
    el("data").textContent = "";
    try {
      if (!frame.data) throw new Error("数据文件未保存或已缺失");
      const response = await fetch(frame.data);
      if (!response.ok) throw new Error("此帧数据读取失败，请刷新图片列表");
      const data = await response.json();
      if (serial !== frameRequest) return;
      el("data").textContent = JSON.stringify(data, null, 2);
      const quality = data.quality || {};
      const rms = quality.reprojection_rms_px;
      const summary = [`角点：${quality.corners ?? "—"}`, `重投影误差：${Number.isFinite(rms) ? rms.toFixed(3)+" px" : "—"}`];
      if (data.time_unix_s) summary.push(`采样时间：${new Date(data.time_unix_s * 1000).toLocaleString()}`);
      if (data.error) summary.push(`检测未通过：${data.error}`);
      el("summary").textContent = summary.join(" · ");
    } catch (error) {
      if (serial === frameRequest) el("summary").textContent = error.message;
    }
  }
  function showSample(previousFrame = "selected") {
    const sample = selected();
    if (!sample) return;
    const index = catalog.samples.indexOf(sample);
    el("prev").disabled = index === 0;
    el("next").disabled = index === catalog.samples.length - 1;
    options(el("frame"), sample.frames.map((f) => [f.id, f.label]), previousFrame);
    showFrame();
  }
  async function refresh() {
    const serial = ++listRequest;
    ++frameRequest;
    el("status").textContent = "正在读取已保存的图片…";
    try {
      const response = await fetch("/api/samples");
      if (!response.ok) throw new Error("图片列表读取失败，请点击刷新重试。");
      const data = await response.json();
      if (serial !== listRequest) return;
      const previous = catalog.key === data.key ? el("sample").value : null;
      const previousFrame = previous ? el("frame").value : "selected";
      catalog = data;
      el("content").hidden = !data.samples.length;
      el("status").textContent = data.samples.length ? `已保存 ${data.samples.length} 组样本，可按组查看。` : "尚无已保存的采样图片。完成采样后点击刷新即可查看。";
      el("source").textContent = data.dataset ? `当前数据集：${data.dataset}` : "";
      options(el("sample"), data.samples.map((s) => [s.id, `第 ${s.number} 组`]), previous);
      showSample(previousFrame);
    } catch (error) {
      if (serial !== listRequest) return;
      el("status").textContent = error.message;
      el("content").hidden = true;
    }
  }
  el("refresh").addEventListener("click", refresh);
  el("sample").addEventListener("change", () => showSample());
  el("frame").addEventListener("change", showFrame);
  el("kind").addEventListener("change", showFrame);
  for (const [id, offset] of [["prev", -1], ["next", 1]]) {
    el(id).addEventListener("click", () => {
      const index = catalog.samples.indexOf(selected()) + offset;
      if (!catalog.samples[index]) return;
      el("sample").value = catalog.samples[index].id;
      showSample();
    });
  }
  window.calibrationGallery = { update(state) {
    const key = `${state.state?.dataset}:${state.job?.id}:${state.job?.status}`;
    if (key !== refreshKey) { refreshKey = key; refresh(); }
  }};
  refresh();
})();
