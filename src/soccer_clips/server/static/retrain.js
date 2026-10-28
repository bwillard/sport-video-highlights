// A model's panel (recognition on the roster page, ball on the ball labels page): what's in use, the runs, and
// buttons that start jobs on RunPod (retrain.py / ball_retrain.py). Polls every 15 s while a job runs.
//   retrainPanel(el, {kind, summary(data) -> html, actions(data) -> [{action, label, confirm}], how(data) -> html,
//                     onDone()})
const retrainWhen = (iso) => iso ? new Date(iso).toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }) : "?";
const retrainDay = (iso) => iso ? new Date(iso).toLocaleDateString([], { month: "short", day: "numeric" }) : "?";
const retrainPct = (x) => x == null ? "?" : `${Math.round(x * 100)}%`;

// bars comparing numbers out of 1: [[label, value, css class], ...]
function retrainMeters(rows, title = "") {
  return `<div class="meters" title="${esc(title)}">${rows.map(([t, v, c]) =>
    `<div class="meter"><span class="lbl">${esc(t)}</span><span class="track"><span class="fill ${c}" style="width:${Math.round((v || 0) * 100)}%"></span></span><span class="v">${retrainPct(v)}</span></div>`).join("")}</div>`;
}

function retrainPanel(el, opts) {
  let data = null, timer = null, busy = false, wasRunning = false;
  const kind = opts.kind;

  async function load() {
    if (busy) return;  // one at a time
    busy = true;
    try { await loadNow(); } finally { busy = false; }
  }
  async function loadNow() {
    try { data = await getJson(`/api/retrain?kind=${kind}`); } catch (e) { el.textContent = String(e.message || e); return; }
    const running = data.status?.state === "running";
    let logText = "";
    if (running || el.querySelector("details[data-k=log][open]")) {
      try { logText = (await getJson(`/api/retrain/log?kind=${kind}&lines=30`)).lines.join("\n"); } catch { /* not yet */ }
    }
    render(logText);
    clearTimeout(timer);
    if (running) timer = setTimeout(load, 15000);
    else if (wasRunning && data.status?.state === "done") opts.onDone?.();
    wasRunning = running;
  }

  function render(logText) {
    const st = data.status, running = st?.state === "running";
    const parts = [opts.summary(data)];
    if (st) {
      const cls = running ? "run" : st.state === "failed" ? "bad" : "";
      const what = st.action === "retrack" ? "Re-tracking" : "Retraining";
      const label = running ? `${what} (${st.step}), started ${retrainWhen(st.started)}`
        : st.state === "failed" ? `The last ${what.toLowerCase().replace(/ing$/, "")} failed (${retrainWhen(st.finished || st.updated)})`
        : `Last ${st.action === "retrack" ? "re-track" : "retrain"} finished ${retrainWhen(st.finished)}`;
      parts.push(`<div class="${cls}">${esc(label)}${st.dollars ? ` · $${st.dollars.toFixed(2)} spent` : ""}${st.message ? `<br><span class="msg">${esc(st.message)}</span>` : ""}</div>`);
    }
    parts.push(data.runpod
      ? opts.actions(data).map((a, i) => `<button data-act="${i}" ${running ? "disabled" : ""}>${running ? "Busy on RunPod…" : a.label}</button>`).join("<br>")
      : `<div>This needs RunPod (RUNPOD_API_KEY in .env).</div>`);
    if (opts.how) parts.push(`<details data-k="how"><summary>How is this measured?</summary><div class="msg">${opts.how(data)}</div></details>`);
    if (st?.log) parts.push(`<details data-k="log" ${running ? "open" : ""}><summary>log</summary><pre>${esc(logText || "…")}</pre></details>`);
    // a redraw keeps each fold-out as the viewer left it
    const open = Object.fromEntries([...el.querySelectorAll("details[data-k]")].map((d) => [d.dataset.k, d.open]));
    el.innerHTML = parts.join("");
    for (const d of el.querySelectorAll("details[data-k]")) if (d.dataset.k in open) d.open = open[d.dataset.k];
    const pre = el.querySelector("pre");
    if (pre) pre.scrollTop = pre.scrollHeight;
  }

  el.addEventListener("click", async (e) => {
    const b = e.target.closest("[data-act]");
    if (b) {
      const a = opts.actions(data)[+b.dataset.act];
      if (!confirm(a.confirm)) return;
      try {
        await api("POST", "/api/retrain", { kind, action: a.action });
        flash("started on RunPod");
      } catch (err) { flash(String(err.message || err), true); }
      load();
      return;
    }
    // opening the log fetches it now; only a click does (redraws open and close fold-outs too, and their toggle
    // events would set off another fetch, and so on)
    const d = e.target.closest("summary")?.parentElement;
    if (d?.dataset.k === "log" && !d.open) setTimeout(load, 0);
  });
  return { load, data: () => data };
}
