(() => {
  "use strict";

  const BOARD_COLORS = ["#00aeef", "#45c982", "#9b68df", "#ffc21c", "#ff8500", "#1478f2"];
  const ICON_SPRITE = "/assets/phosphor-icons.svg";
  const ROUTES = { "/": "monitor", "/graph": "monitor", "/spectrum": "spectrum", "/replay": "replay", "/ask": "ask", "/chat": "ask" };
  const state = {
    view: ROUTES[location.pathname] || "monitor",
    sensors: [],
    baselineId: "baseline",
    readings: new Map(),
    monitor: null,
    gemmaTargets: [],
    syntheticInjection: null,
    syntheticScheduled: false,
    offlineReplay: null,
    paused: false,
    monitorAxis: "norm",
    spectrumAxis: "norm",
    spectrum: null,
    spectrumRange: [0, 0],
    replays: [],
    replayIndex: -1,
    replay: null,
    context: null,
    history: [],
  };

  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const finite = (value) => Number.isFinite(Number(value)) ? Number(value) : null;
  const formatNumber = (value, digits = 3) => {
    const number = finite(value);
    if (number === null) return "-";
    if (number !== 0 && (Math.abs(number) < 0.001 || Math.abs(number) >= 10000)) return number.toExponential(2);
    return number.toLocaleString(undefined, { maximumFractionDigits: digits });
  };
  const formatAge = (seconds) => {
    const value = finite(seconds);
    if (value === null) return "-";
    if (value < 2) return "Just now";
    if (value < 60) return `${Math.round(value)} s ago`;
    return `${Math.round(value / 60)} min ago`;
  };
  const formatDate = (value) => {
    const date = new Date(value || "");
    return Number.isNaN(date.getTime()) ? "-" : date.toLocaleString([], { dateStyle: "medium", timeStyle: "medium" });
  };
  const boardLabel = (id) => id === state.baselineId ? "Reference" : String(id || "").replace("target_", "Target ");
  const boardShort = (id) => id === state.baselineId ? "R" : String(id || "").replace("target_", "T");
  const formatSig = (value, digits = 3) => {
    const number = finite(value);
    if (number === null) return "-";
    if (number === 0) return "0";
    if (Math.abs(number) < 1e-4 || Math.abs(number) >= 1e5) return number.toExponential(Math.max(1, digits - 1));
    return String(Number(number.toPrecision(digits)));
  };
  const readoutStats = (stats) => {
    if (!stats) return null;
    const mean = finite(stats.mean_g); const rms = finite(stats.rms_g); const min = finite(stats.min_g); const max = finite(stats.max_g);
    if (mean === null || rms === null) return null;
    const acRms = Math.sqrt(Math.max(0, rms * rms - mean * mean));
    const peak = Math.max(max === null ? 0 : max - mean, min === null ? 0 : mean - min);
    return { acRms, peak };
  };
  function renderRibbon(container, cells) {
    if (!container) return;
    container.replaceChildren();
    cells.forEach((cell) => {
      const item = document.createElement("span");
      item.className = `ribbon-cell ${cell.kind || "normal"}`;
      item.style.setProperty("--board-color", cell.color);
      item.title = `${boardLabel(cell.sensor_id)} · ${cell.label}`;
      const bar = document.createElement("i");
      const label = document.createElement("b"); label.textContent = boardShort(cell.sensor_id);
      item.append(bar, label); container.append(item);
    });
  }
  const setIcon = (use, name) => use?.setAttribute("href", `${ICON_SPRITE}#ph-${name}`);
  const makeIcon = (name, className = "ph-icon") => {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", className); svg.setAttribute("aria-hidden", "true");
    const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    setIcon(use, name); svg.append(use); return svg;
  };

  async function api(path, options) {
    const response = await fetch(path, { cache: "no-store", ...options });
    let payload;
    try { payload = await response.json(); } catch { payload = {}; }
    if (!response.ok || payload.ok === false) throw new Error(payload.message || payload.error || `Request failed (${response.status})`);
    return payload;
  }

  function configureView() {
    $$('[data-view]').forEach((view) => { view.hidden = view.dataset.view !== state.view; });
    $$('[data-route]').forEach((link) => link.classList.toggle("active", link.dataset.route === state.view));
  }

  function setLiveState(ok, message) {
    const element = $("#liveState");
    element.classList.toggle("offline", !ok);
    element.classList.toggle("waiting", message === "Connecting");
    element.querySelector("strong").textContent = message;
  }

  async function loadSensors() {
    try {
      const payload = await api("/api/vibro/sensors?process_reader=1&prewarm=1");
      state.sensors = Array.isArray(payload.sensors) ? payload.sensors : [];
      state.baselineId = payload.baseline_sensor_id || "baseline";
      state.offlineReplay = payload.replay || null;
      const monitorLabel = state.offlineReplay ? "Offline monitor" : "Live monitor";
      $("#monitorTitle").textContent = monitorLabel;
      $("[data-route='monitor']").textContent = monitorLabel;
      setLiveState(
        state.sensors.length === 6,
        state.offlineReplay ? `Offline replay · ${state.sensors.length}/6 recordings` : `Live · ${state.sensors.length}/6 boards`,
      );
    } catch (error) {
      state.sensors = [];
      setLiveState(false, "Boards unavailable");
      console.warn(error);
    }
    renderSensorControls();
    renderMonitorStatus();
  }

  function renderMonitorStatus() {
    const status = $("#monitorStatus"); if (!status) return;
    if (!state.sensors.length) { status.textContent = "No boards connected · check the acquisition folders"; return; }
    const axis = state.monitorAxis === "norm" ? "norm" : state.monitorAxis.toUpperCase();
    const firstReading = state.readings.values().next().value;
    const rate = finite(firstReading?.sampling_rate_hz);
    const fixture = String(firstReading?.metadata?.source_fixture || "").split("/")[0];
    const overlay = firstReading?.metadata?.waveform_source === "live_with_lumo_overlay";
    const replayPosition = finite(state.offlineReplay?.position_s);
    const source = state.offlineReplay
      ? `Offline replay${replayPosition === null ? "" : ` · ${formatNumber(replayPosition, 1)} s`}`
      : `${state.sensors.length}/6 boards`;
    const parts = [overlay ? `${state.offlineReplay ? "Offline" : "Live"} + LUMO ${fixture}` : source, `${axis} axis`, `${$("#monitorWindow")?.value || "10"} s window`];
    if (rate) parts.push(`${formatSig(rate / 1000, 3)} kHz`);
    parts.push(state.paused ? "paused" : "2 s refresh");
    status.textContent = parts.map((part) => part.replace(/ /g, "\u00a0")).join(" · ");
  }

  function renderSensorControls() {
    if (state.view === "monitor") renderMonitorScaffold();
    const select = $("#spectrumBoard");
    if (!select) return;
    select.replaceChildren();
    state.sensors.forEach((sensor, index) => {
      const option = document.createElement("option");
      option.value = sensor.sensor_id;
      option.textContent = boardLabel(sensor.sensor_id);
      option.dataset.index = String(index);
      select.append(option);
    });
    const requested = new URLSearchParams(location.search).get("board");
    if (requested && state.sensors.some((sensor) => sensor.sensor_id === requested)) select.value = requested;
    updateSpectrumBoardDot();
  }

  function updateSpectrumBoardDot() {
    const index = Math.max(0, state.sensors.findIndex((sensor) => sensor.sensor_id === $("#spectrumBoard")?.value));
    const dot = $(".select-board-dot"); if (dot) dot.style.background = BOARD_COLORS[index % BOARD_COLORS.length];
  }

  function isSharedTargetPattern(targets) {
    if (!Array.isArray(targets) || targets.length !== 5 || !targets.every((target) => target.affected)) return false;
    return targets.every((target) => target.class === targets[0].class && target.severity === targets[0].severity);
  }

  function classLabel(value) {
    return String(value || "Anomaly").split("_").map((word) => word.charAt(0).toUpperCase() + word.slice(1)).join(" ");
  }

  function targetState(sensorId) {
    if (sensorId === state.baselineId) return { label: "Live", kind: "normal" };
    const target = state.gemmaTargets.find((item) => item.sensor_id === sensorId);
    if (!target) return { label: "Waiting", kind: "unavailable" };
    if (target.class === "data_invalid") return { label: "Data invalid", kind: "quality" };
    if (target.localization_candidate) return { label: "Anomaly candidate", kind: "shared" };
    if (isSharedTargetPattern(state.gemmaTargets)) return { label: classLabel(target.class), kind: "shared" };
    return target.affected ? { label: classLabel(target.class), kind: "anomaly" } : { label: "Normal", kind: "normal" };
  }

  function renderMonitorScaffold() {
    const stack = $("#waveformStack");
    const boardStates = $("#boardStates");
    if (!stack || !boardStates) return;
    stack.replaceChildren();
    boardStates.replaceChildren();
    state.sensors.forEach((sensor, index) => {
      const color = BOARD_COLORS[index % BOARD_COLORS.length];
      const row = document.createElement("article");
      row.className = "wave-row";
      row.style.setProperty("--board-color", color);
      row.dataset.sensorId = sensor.sensor_id;
      const info = document.createElement("div");
      info.className = "wave-info";
      const title = document.createElement("strong");
      title.textContent = boardLabel(sensor.sensor_id);
      const status = document.createElement("span");
      const mapped = targetState(sensor.sensor_id);
      row.classList.add(mapped.kind);
      status.className = `wave-status ${mapped.kind}`;
      status.textContent = mapped.label;
      const readout = document.createElement("span");
      readout.className = "wave-readout";
      readout.title = "Mean-removed RMS and peak deviation over the last window";
      const rmsLine = document.createElement("span"); rmsLine.textContent = "reading…";
      const peakLine = document.createElement("span");
      readout.append(rmsLine, peakLine);
      info.append(title, status, readout);
      const canvasWrap = document.createElement("div");
      canvasWrap.className = "wave-canvas-wrap";
      const canvas = document.createElement("canvas");
      canvas.width = 900;
      canvas.height = 86;
      canvas.setAttribute("aria-label", `${boardLabel(sensor.sensor_id)} ${state.offlineReplay ? "recorded" : "live"} waveform`);
      canvasWrap.append(canvas);
      row.append(info, canvasWrap);
      stack.append(row);

      const stateRow = document.createElement("div");
      stateRow.className = "state-row";
      stateRow.style.setProperty("--board-color", color);
      const dot = document.createElement("span"); dot.className = "state-dot";
      const name = document.createElement("span"); name.textContent = boardLabel(sensor.sensor_id);
      const value = document.createElement("span"); value.className = `state-value ${mapped.kind}`; value.textContent = mapped.label;
      stateRow.append(dot, name, value);
      boardStates.append(stateRow);
    });
    renderRibbon($("#verdictRibbon"), state.sensors.map((sensor, index) => {
      const mapped = targetState(sensor.sensor_id);
      return { sensor_id: sensor.sensor_id, color: BOARD_COLORS[index % BOARD_COLORS.length], kind: sensor.sensor_id === state.baselineId ? "normal" : mapped.kind, label: mapped.label };
    }));
    redrawWaveforms();
  }

  function waveformUrl(sensor) {
    const params = new URLSearchParams({
      machine_id: sensor.sensor_id,
      sensor_name: sensor.hsd_sensor_name || "iis3dwb_acc",
      axis: state.monitorAxis,
      duration_s: $("#monitorWindow")?.value || "10",
      max_points: "720",
      require_current: "1",
      process_reader: "1",
    });
    if (sensor.acquisition_folder) params.set("acquisition_folder", sensor.acquisition_folder);
    return `/api/vibro/waveform?${params}`;
  }

  async function fetchWaveforms() {
    if (state.paused || state.view !== "monitor" || !state.sensors.length) return;
    const results = await Promise.all(state.sensors.map(async (sensor) => {
      try { return [sensor.sensor_id, await api(waveformUrl(sensor))]; }
      catch (error) { return [sensor.sensor_id, { error: error.message }]; }
    }));
    const replay = results.find(([, payload]) => payload?.replay)?.[1]?.replay;
    if (replay) state.offlineReplay = replay;
    results.forEach(([id, payload]) => state.readings.set(id, payload));
    redrawWaveforms();
  }

  function waveformAmplitude(payload) {
    const samples = Array.isArray(payload?.samples_g) ? payload.samples_g.map(Number).filter(Number.isFinite) : [];
    if (!samples.length) return 0;
    const mean = samples.reduce((sum, value) => sum + value, 0) / samples.length;
    return Math.max(...samples.map((value) => Math.abs(value - mean)));
  }

  function drawWaveform(canvas, payload, color, sharedMaxAbs) {
    const rect = canvas.getBoundingClientRect();
    const ratio = Math.max(1, Math.min(2, devicePixelRatio || 1));
    const width = Math.max(260, Math.round(rect.width * ratio));
    const height = Math.max(70, Math.round(rect.height * ratio));
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, width, height);
    const samples = Array.isArray(payload?.samples_g) ? payload.samples_g.map(Number).filter(Number.isFinite) : [];
    ctx.strokeStyle = "#e6ecf3"; ctx.lineWidth = ratio;
    for (let i = 1; i < 4; i += 1) { const y = height * i / 4; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke(); }
    for (let i = 1; i < 5; i += 1) { const x = Math.round(width * i / 5) + .5; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke(); }
    ctx.strokeStyle = "#d3dce7"; ctx.beginPath(); ctx.moveTo(0, Math.round(height / 2) + .5); ctx.lineTo(width, Math.round(height / 2) + .5); ctx.stroke();
    if (!samples.length) return;
    const mean = samples.reduce((sum, value) => sum + value, 0) / samples.length;
    const centered = samples.map((value) => value - mean);
    let maxAbs = Number(sharedMaxAbs);
    if (!Number.isFinite(maxAbs) || maxAbs < 1e-7) maxAbs = 1e-7;
    ctx.strokeStyle = color; ctx.lineWidth = 1.25 * ratio; ctx.beginPath();
    centered.forEach((value, index) => {
      const x = index / Math.max(1, centered.length - 1) * width;
      const y = height / 2 - value / maxAbs * height * .38;
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  function redrawWaveforms() {
    const sharedMaxAbs = Math.max(1e-7, ...state.sensors.map((sensor) => waveformAmplitude(state.readings.get(sensor.sensor_id))));
    const hasData = state.sensors.some((sensor) => Array.isArray(state.readings.get(sensor.sensor_id)?.samples_g));
    const scale = $("#scaleReadout");
    if (scale) scale.textContent = hasData ? `Shared scale ±${formatSig(sharedMaxAbs, 3)} g` : "Shared scale · waiting for data";
    $$(".wave-row").forEach((row, index) => {
      const canvas = row.querySelector("canvas");
      const wrapper = row.querySelector(".wave-canvas-wrap");
      wrapper.querySelector(".wave-error")?.remove();
      wrapper.querySelector(".wave-source")?.remove();
      const payload = state.readings.get(row.dataset.sensorId);
      wrapper.title = `Shared vertical scale: ±${formatSig(sharedMaxAbs, 3)} g`;
      drawWaveform(canvas, payload, BOARD_COLORS[index % BOARD_COLORS.length], sharedMaxAbs);
      const readout = row.querySelector(".wave-readout");
      if (readout) {
        const stats = readoutStats(payload?.stats);
        readout.children[0].textContent = stats ? `RMS ${formatSig(stats.acRms, 3)} g` : (payload?.error ? `no ${state.offlineReplay ? "replay" : "live"} data` : "reading…");
        readout.children[1].textContent = stats ? `peak ${formatSig(stats.peak, 3)} g` : "";
      }
      if (payload?.error) {
        const error = document.createElement("span"); error.className = "wave-error"; error.textContent = state.offlineReplay ? "Replay data unavailable" : "Live data unavailable"; wrapper.append(error);
      } else if (payload?.metadata?.waveform_source === "live_with_lumo_overlay") {
        const source = document.createElement("span"); source.className = "wave-source"; source.textContent = `${state.offlineReplay ? "Offline" : "Live"} + LUMO test overlay`; wrapper.append(source);
      }
    });
    renderMonitorStatus();
  }

  function monitorContext(payload) {
    const result = payload?.result || {};
    const meta = result.model_metadata || {};
    const monitor = meta.vibrogemma_monitor || {};
    const targets = Array.isArray(monitor.targets) ? monitor.targets : [];
    const sharedPattern = isSharedTargetPattern(targets);
    const globalClass = String(monitor.global_class || "");
    const globalKnown = Boolean(globalClass);
    const globalAnomaly = globalKnown && !["normal", "data_invalid"].includes(globalClass);
    const targetAnomaly = targets.some((target) => target.affected);
    const quality = globalClass === "data_invalid" || targets.some((target) => target.class === "data_invalid");
    const inferredConsistent = globalKnown ? globalAnomaly === targetAnomaly : null;
    const globalTargetConsistent = typeof monitor.global_target_consistent === "boolean" ? monitor.global_target_consistent : inferredConsistent;
    const inconclusive = !globalAnomaly && (globalTargetConsistent === false || (sharedPattern && !globalKnown));
    const reportedLocalization = String(monitor.localization_status || "");
    const geometryUnavailable = monitor.localization_verified === false || reportedLocalization === "geometry_unavailable";
    const localizationStatus = globalAnomaly && !targetAnomaly
      ? "unresolved_global_anomaly"
      : (inconclusive ? "inconclusive" : (geometryUnavailable && targetAnomaly ? "geometry_unavailable" : (sharedPattern ? "network_wide" : (targetAnomaly ? "target" : "normal"))));
    const deployment = payload?.deployment || {};
    let explanation = result.main_agent_explanation || "";
    if (inconclusive) explanation = "The global and target decisions disagree, so this check is inconclusive.";
    else if (globalAnomaly && !targetAnomaly) explanation = "The Agent detected a global anomaly but did not localize it to a target.";
    else if (globalAnomaly && sharedPattern) explanation = geometryUnavailable
      ? "The Agent detected an anomaly and marked all five targets as model candidates; physical localization is unavailable."
      : "The Agent detected a network-wide anomaly and marked all five targets; localization is not isolated.";
    else if (geometryUnavailable && targetAnomaly) explanation = `The Agent marked ${targets.filter((target) => target.affected).map((target) => boardLabel(target.sensor_id)).join(", ")} as model candidates, but physical localization is unavailable because sensor geometry is not configured.`;
    return {
      source: "latest_gemma_check",
      available: Boolean(!payload?.stale && meta.monitor_llm_model_used && targets.length === 5),
      stale: Boolean(payload?.stale),
      network_state: quality ? "data_invalid" : (globalAnomaly ? "anomaly" : (inconclusive ? "inconclusive" : (targetAnomaly ? "anomaly" : "normal"))),
      shared_pattern: sharedPattern,
      global_only: globalAnomaly && !targetAnomaly,
      inconclusive,
      global_class: globalClass || null,
      global_target_consistent: globalTargetConsistent,
      localization_status: localizationStatus,
      localization_verified: geometryUnavailable ? false : monitor.localization_verified,
      targets: targets.map(({ sensor_id, affected, class: className, severity }) => ({ sensor_id, affected, class: className, severity, localization_candidate: Boolean((geometryUnavailable || inconclusive) && affected) })),
      explanation,
      soft_tokens_valid: monitor.soft_tokens_valid,
      updated_age_s: finite(payload?.diagnostics?.cached_age_s),
      window_duration_s: 10,
      deployment,
      research_only: String(deployment.promotion_status || "").includes("unpromoted") || String(deployment.permitted_use || "").includes("research"),
    };
  }

  async function fetchMonitor() {
    const params = new URLSearchParams({ axis: "norm", duration_s: "10", require_current: "1", require_llm: "0", use_llm: "1", model_timeout_s: "60", vote_k: "1", skip_cache: "1", process_reader: "1" });
    try {
      const payload = await api(`/api/vibro/monitor?${params}`);
      state.monitor = payload;
      const context = monitorContext(payload);
      state.gemmaTargets = context.available ? context.targets : [];
      if (!state.context || state.context.source === "latest_gemma_check") state.context = context;
      if (state.view === "monitor") renderMonitorResult(context, payload.popup || {});
      if (state.view === "ask" && state.context?.source === "latest_gemma_check") renderContext();
    } catch (error) {
      console.warn(error);
      const unavailable = { source: "latest_gemma_check", available: false, stale: false, targets: [] };
      state.monitor = null;
      state.gemmaTargets = [];
      if (!state.context || state.context.source === "latest_gemma_check") state.context = unavailable;
      hideAlert();
      if (state.view === "monitor") renderMonitorResult(unavailable, {});
      if (state.view === "ask" && state.context?.source === "latest_gemma_check") renderContext();
    }
  }

  function renderMonitorResult(context, popup) {
    const verdict = $("#networkVerdict");
    const verdictIcon = $("#verdictIcon");
    const researchNote = context.research_only ? " · Research/replay-only checkpoint" : "";
    const coverage = `${formatNumber(context.window_duration_s || 10, 0)} s`;
    const age = formatAge(context.updated_age_s).toLowerCase();
    const title = $("#networkTitle"); const detail = $("#verdictDetail");
    verdict.className = "verdict-card";
    if (!context.available) {
      verdict.classList.add("quality");
      title.textContent = context.stale ? "Agent check stale" : "Agent unavailable";
      setIcon(verdictIcon, "clock");
      detail.textContent = context.stale
        ? `Last analyzed window is ${age}; verdict withheld until a fresh check completes.${researchNote}`
        : `Live waveforms remain available.${researchNote}`;
    } else if (context.network_state === "normal") {
      verdict.classList.add("normal"); title.textContent = "No anomaly detected";
      setIcon(verdictIcon, "check-circle"); detail.textContent = `Latest analyzed ${coverage} window · updated ${age}${researchNote}`;
    } else if (context.network_state === "data_invalid") {
      verdict.classList.add("quality", "data-invalid"); title.textContent = "Data invalid";
      setIcon(verdictIcon, "shield"); detail.textContent = `Signal quality blocked a verdict · updated ${age}${researchNote}`;
    } else if (context.network_state === "inconclusive") {
      verdict.classList.add("quality", "shared"); title.textContent = "Normal not confirmed";
      setIcon(verdictIcon, "shield"); detail.textContent = context.global_target_consistent === false
        ? `Global and target verdicts disagree · updated ${age}${researchNote}`
        : `Target localization unavailable · updated ${age}${researchNote}`;
    } else {
      verdict.classList.add("anomaly"); title.textContent = context.global_only ? "Network anomaly" : "Anomaly detected";
      setIcon(verdictIcon, "warning-circle"); detail.textContent = context.global_only
        ? `Localization unavailable · updated ${age}${researchNote}`
        : (context.localization_status === "geometry_unavailable"
          ? `Physical localization unavailable · updated ${age}${researchNote}`
          : `One or more targets are affected · updated ${age}${researchNote}`);
    }
    $("#tokenCount").textContent = context.available ? `${context.soft_tokens_valid || 84}` : "-";
    $("#checkAge").textContent = formatAge(context.updated_age_s);
    $("#affectedCount").textContent = context.available ? (context.localization_status === "geometry_unavailable" ? "Unverified" : (context.shared_pattern ? "Unlocalized" : String(context.targets.filter((target) => target.affected).length))) : "-";
    renderMonitorScaffold();
    const alertActive = Boolean(context.available && context.network_state !== "normal" && popup.show_popup);
    $("#alertTray").classList.toggle("active", alertActive);
    $("#alertTray").querySelector("strong").textContent = alertActive ? "Active Agent alert" : "No active alerts";
    if (alertActive) showAlert(context); else hideAlert();
  }

  function showAlert(context) {
    const toast = $("#alertToast");
    const affected = context.targets.filter((target) => target.affected);
    const quality = context.targets.filter((target) => target.class === "data_invalid");
    const sharedPattern = context.shared_pattern;
    const conflict = context.inconclusive;
    const geometryUnavailable = context.localization_status === "geometry_unavailable";
    toast.classList.toggle("quality", quality.length > 0 || conflict);
    $("#alertKicker").textContent = quality.length ? "Agent data quality" : (conflict ? "Agent review required" : "Agent advisory");
    let alertTitle = `${affected.length} target${affected.length === 1 ? "" : "s"} need review`;
    let alertSummary = `Agent found ${affected.length === 1 ? "a difference" : "differences"} from the reference in the current 10 s window.`;
    if (quality.length) {
      alertTitle = "Window needs attention";
      alertSummary = "Signal quality prevented a reliable classification for this window.";
    } else if (conflict) {
      alertTitle = "Conflicting decisions";
      alertSummary = "Global and target decisions conflict; review required. This is not a localized anomaly.";
    } else if (context.global_only) {
      alertTitle = "Network anomaly";
      alertSummary = "The global decision is anomalous, but no target was localized.";
    } else if (geometryUnavailable) {
      alertTitle = "Anomaly; localization unverified";
      alertSummary = "Target IDs are model candidates; physical localization is unavailable because sensor positions are not configured.";
    } else if (sharedPattern) {
      alertTitle = "Network-level advisory";
      alertSummary = "Agent returned the same state for all five targets. This result is not localized, so it must not be read as five independent faults.";
    }
    $("#alertTitle").textContent = alertTitle;
    $("#alertSummary").textContent = alertSummary;
    const targets = $("#alertTargets");
    targets.replaceChildren();
    const items = quality.length ? quality : affected;
    if ((sharedPattern || context.global_only || conflict) && !quality.length) {
      const item = document.createElement("span");
      item.className = "alert-target shared";
      item.append(makeIcon("waveform"), document.createTextNode("Localization unavailable"));
      targets.append(item);
    } else {
      items.forEach((target) => {
        const item = document.createElement("span");
        item.className = "alert-target";
        item.textContent = geometryUnavailable
            ? `${boardLabel(target.sensor_id)} · Model candidate`
            : `${boardLabel(target.sensor_id)} · ${classLabel(target.class)}`;
        targets.append(item);
      });
    }
    $("#alertTokenMeta").textContent = `${context.soft_tokens_valid || 84} continuous tokens`;
    $("#alertAgeMeta").textContent = `Updated ${formatAge(context.updated_age_s).toLowerCase()}`;
    toast.hidden = false;
  }
  function hideAlert() { $("#alertToast").hidden = true; }

  function renderSyntheticControl() {
    if (state.syntheticScheduled) {
      $("#toggleSynthetic").disabled = true;
      $("#toggleSyntheticTarget5").disabled = true;
      $("#syntheticActionLabel").textContent = "Target 3 LUMO event scheduled";
      $("#syntheticTarget5ActionLabel").textContent = "Target 5 LUMO event scheduled";
      return;
    }
    const target3 = state.syntheticInjection === "target_3";
    const target5 = state.syntheticInjection === "target_5";
    $("#toggleSynthetic")?.setAttribute("aria-pressed", String(target3));
    $("#toggleSyntheticTarget5")?.setAttribute("aria-pressed", String(target5));
    $("#syntheticActionLabel").textContent = target3 ? "Stop Target 3 anomaly test" : "Run Target 3 anomaly test";
    $("#syntheticTarget5ActionLabel").textContent = target5 ? "Stop Target 5 anomaly test" : "Run Target 5 anomaly test";
  }

  async function loadSyntheticInjection() {
    const button = $("#toggleSynthetic");
    try {
      const payload = await api("/api/vibro/synthetic-injection");
      state.syntheticInjection = payload.enabled ? payload.target : null;
      state.syntheticScheduled = Boolean(payload.scheduled);
      renderSyntheticControl();
    } catch (error) {
      console.warn(error);
      if (button) button.disabled = true;
      if ($("#toggleSyntheticTarget5")) $("#toggleSyntheticTarget5").disabled = true;
      $("#syntheticActionLabel").textContent = "Injection unavailable";
      $("#syntheticTarget5ActionLabel").textContent = "Injection unavailable";
    }
  }

  async function toggleSyntheticInjection(target) {
    const buttons = [$("#toggleSynthetic"), $("#toggleSyntheticTarget5")].filter(Boolean);
    buttons.forEach((button) => { button.disabled = true; });
    try {
      const payload = await api("/api/vibro/synthetic-injection", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: state.syntheticInjection !== target, target }),
      });
      state.syntheticInjection = payload.enabled ? payload.target : null;
      renderSyntheticControl();
    } catch (error) {
      console.warn(error);
    } finally {
      buttons.forEach((button) => { button.disabled = false; });
    }
  }

  function setupSegments(container, onChange) {
    container?.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-value]");
      if (!button) return;
      container.querySelectorAll("button").forEach((item) => item.classList.toggle("selected", item === button));
      onChange(button.dataset.value);
    });
  }

  function setupMonitor() {
    const updateTimeRuler = () => {
      const duration = Number($("#monitorWindow")?.value) || 10;
      $$(".time-ruler span").forEach((span, index, labels) => {
        const seconds = duration * (labels.length - 1 - index) / (labels.length - 1);
        span.textContent = seconds ? `-${seconds.toFixed(duration <= 2 ? 1 : 0)} s` : "0 s";
      });
    };
    setupSegments($("#monitorAxis"), (value) => { state.monitorAxis = value; state.readings.clear(); renderMonitorStatus(); fetchWaveforms(); });
    updateTimeRuler();
    $("#monitorWindow")?.addEventListener("change", () => { updateTimeRuler(); state.readings.clear(); renderMonitorStatus(); fetchWaveforms(); });
    $("#pauseMonitor")?.addEventListener("click", () => {
      state.paused = !state.paused;
      const button = $("#pauseMonitor");
      button.setAttribute("aria-pressed", String(state.paused));
      button.querySelector("span").textContent = state.paused ? "Resume" : "Pause";
      setIcon(button.querySelector("use"), state.paused ? "play" : "pause");
      renderMonitorStatus();
      if (!state.paused) fetchWaveforms();
    });
    $("#askLatest")?.addEventListener("click", () => storeContext(state.context));
    $("#toggleSynthetic")?.addEventListener("click", () => toggleSyntheticInjection("target_3"));
    $("#toggleSyntheticTarget5")?.addEventListener("click", () => toggleSyntheticInjection("target_5"));
    $("#closeToast")?.addEventListener("click", hideAlert);
    $("#alertTray")?.addEventListener("click", () => {
      if (state.monitor?.popup?.show_popup) showAlert(monitorContext(state.monitor));
    });
    window.addEventListener("resize", redrawWaveforms);
  }

  function selectedSensor() { return state.sensors.find((sensor) => sensor.sensor_id === $("#spectrumBoard")?.value) || state.sensors[0]; }
  function spectrumUrl() {
    const sensor = selectedSensor();
    const params = new URLSearchParams({
      machine_id: sensor.sensor_id,
      sensor_name: sensor.hsd_sensor_name || "iis3dwb_acc",
      axis: state.spectrumAxis,
      duration_s: $("#spectrumWindow").value,
      estimator: $("#spectrumEstimator").value,
      window: "hann",
      plot_scale: "log",
      max_bins: "1500",
      waveform_max_points: "240",
      require_current: "1",
      process_reader: "1",
    });
    if (sensor.acquisition_folder) params.set("acquisition_folder", sensor.acquisition_folder);
    return `/api/vibro/spectrum?${params}`;
  }

  function axisName(axis) { return axis === "norm" ? "norm" : String(axis || "").toUpperCase(); }

  async function runSpectrum() {
    const button = $("#runSpectrum"); const label = button.querySelector("span");
    const sensor = selectedSensor(); if (!sensor) return;
    button.disabled = true; label.textContent = "Running…";
    $("#spectrumStatus").textContent = `Reading ${boardLabel(sensor.sensor_id)} · ${axisName(state.spectrumAxis)} axis · ${$("#spectrumWindow").value} s window…`;
    $("#spectrumEmpty").hidden = false; $("#spectrumEmpty").textContent = "Reading the selected board and computing its spectrum…";
    try {
      state.spectrum = await api(spectrumUrl());
      renderSpectrum();
      const data = state.spectrum;
      const estimatorLabel = $("#spectrumEstimator")?.selectedOptions?.[0]?.textContent || data.estimator_label || data.estimator;
      $("#spectrumStatus").textContent = [
        boardLabel(sensor.sensor_id), `${axisName(state.spectrumAxis)} axis`, estimatorLabel,
        `${formatSig(data.window_duration_s, 3)} s window`, data.segment_count ? `${data.segment_count} segments` : null,
        data.fft_resolution_hz ? `${formatSig(data.fft_resolution_hz, 3)} Hz bins` : null,
      ].filter(Boolean).map((part) => part.replace(/ /g, "\u00a0")).join(" · ");
    } catch (error) {
      state.spectrum = null; drawSpectrum();
      $("#spectrumStatus").textContent = "Spectrum failed";
      $("#spectrumEmpty").hidden = false; $("#spectrumEmpty").textContent = error.message;
    } finally {
      button.disabled = false; label.textContent = "Run spectrum";
    }
  }

  function drawSpectrum() {
    const canvas = $("#spectrumCanvas");
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect(); const ratio = Math.max(1, Math.min(2, devicePixelRatio || 1));
    const width = Math.max(500, Math.round(rect.width * ratio)); const height = Math.max(360, Math.round(rect.height * ratio));
    canvas.width = width; canvas.height = height;
    const ctx = canvas.getContext("2d"); ctx.clearRect(0, 0, width, height);
    const pad = { l: 64 * ratio, r: 18 * ratio, t: 18 * ratio, b: 40 * ratio };
    const plotW = width - pad.l - pad.r; const plotH = height - pad.t - pad.b;
    const monoFont = `${11 * ratio}px "Noto Sans Mono", Menlo, Consolas, monospace`;
    ctx.strokeStyle = "#e6ecf3"; ctx.lineWidth = ratio;
    for (let i = 0; i <= 6; i += 1) { const y = Math.round(pad.t + plotH * i / 6) + .5; ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(width - pad.r, y); ctx.stroke(); }
    if (!state.spectrum) return;
    const spectrum = state.spectrum;
    const envelopeReady = Array.isArray(spectrum.peak_envelope_frequency_hz) && Array.isArray(spectrum.peak_envelope_psd_g2_per_hz) && spectrum.peak_envelope_frequency_hz.length === spectrum.peak_envelope_psd_g2_per_hz.length && spectrum.peak_envelope_frequency_hz.length > 1;
    const frequencies = envelopeReady ? spectrum.peak_envelope_frequency_hz : (spectrum.frequency_hz || []);
    const powers = envelopeReady ? spectrum.peak_envelope_psd_g2_per_hz : (spectrum.psd_g2_per_hz || []);
    const [rangeMin, rangeMax] = state.spectrumRange;
    const inRange = (frequency) => Number.isFinite(frequency) && frequency > 0 && (!rangeMin || frequency >= rangeMin) && (!rangeMax || frequency <= rangeMax);
    const pairs = frequencies.map((frequency, index) => [Number(frequency), Number(powers[index])]).filter(([frequency, power]) => inRange(frequency) && Number.isFinite(power) && power > 0);
    if (pairs.length < 2) return;
    const minF = Math.max(0.1, pairs[0][0]); const maxF = pairs[pairs.length - 1][0];
    const logs = pairs.map(([, power]) => Math.log10(power)); let minP = Math.min(...logs); let maxP = Math.max(...logs);
    if (maxP - minP < 1) { minP -= .5; maxP += .5; }
    maxP += (maxP - minP) * .08;
    const xAt = (frequency) => pad.l + (Math.log10(frequency) - Math.log10(minF)) / (Math.log10(maxF) - Math.log10(minF)) * plotW;
    const yAt = (power) => pad.t + (maxP - Math.log10(power)) / (maxP - minP) * plotH;
    ctx.font = monoFont; ctx.fillStyle = "#66788f"; ctx.textAlign = "center"; ctx.textBaseline = "top";
    [0.1, 1, 10, 100, 1000, 10000].filter((tick) => tick >= minF && tick <= maxF).forEach((tick) => {
      const x = Math.round(xAt(tick)) + .5;
      ctx.strokeStyle = "#e6ecf3"; ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, height - pad.b); ctx.stroke();
      ctx.fillText(tick >= 1000 ? `${tick / 1000}k` : String(tick), x, height - pad.b + 8 * ratio);
    });
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (let i = 0; i <= 6; i += 1) { const exponent = maxP - (maxP - minP) * i / 6; ctx.fillText((10 ** exponent).toExponential(1), pad.l - 8 * ratio, pad.t + plotH * i / 6); }
    const baseline = height - pad.b;
    ctx.beginPath(); ctx.moveTo(xAt(pairs[0][0]), baseline);
    pairs.forEach(([frequency, power]) => ctx.lineTo(xAt(frequency), yAt(power)));
    ctx.lineTo(xAt(pairs[pairs.length - 1][0]), baseline); ctx.closePath();
    ctx.fillStyle = "rgba(0, 115, 198, .08)"; ctx.fill();
    ctx.strokeStyle = "#0073c6"; ctx.lineWidth = 1.5 * ratio; ctx.lineJoin = "round"; ctx.beginPath();
    pairs.forEach(([frequency, power], index) => { const x = xAt(frequency); const y = yAt(power); if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }); ctx.stroke();
    const peaks = (state.spectrum.top_peaks || []).map((peak) => [Number(peak.frequency_hz), Number(peak.psd_g2_per_hz)]).filter(([frequency, power]) => inRange(frequency) && Number.isFinite(power) && power > 0).slice(0, 3);
    const placed = [];
    ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"; ctx.font = `700 ${11 * ratio}px "Noto Sans Mono", Menlo, Consolas, monospace`;
    peaks.forEach(([frequency, power], rank) => {
      const x = xAt(frequency); const y = yAt(power);
      ctx.fillStyle = "#fff"; ctx.strokeStyle = "#03234b"; ctx.lineWidth = 1.5 * ratio;
      ctx.beginPath(); ctx.arc(x, y, 3.5 * ratio, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      const text = `${formatSig(frequency, 4)} Hz`; const textW = ctx.measureText(text).width;
      let labelX = x + 8 * ratio; if (labelX + textW > width - pad.r) labelX = x - 8 * ratio - textW;
      let labelY = y - 8 * ratio;
      while (placed.some(([px, py]) => Math.abs(px - labelX) < textW + 12 * ratio && Math.abs(py - labelY) < 14 * ratio)) labelY -= 15 * ratio;
      if (labelY < pad.t + 10 * ratio) labelY = y + 16 * ratio;
      placed.push([labelX, labelY]);
      ctx.fillStyle = rank === 0 ? "#03234b" : "#4d6280"; ctx.fillText(text, labelX, labelY);
    });
  }

  function renderSpectrum() {
    $("#spectrumEmpty").hidden = true; drawSpectrum();
    const data = state.spectrum;
    $("#spectrumRms").textContent = `${formatSig(data.psd_rms_g, 4)} g`;
    $("#spectrumVariance").textContent = `${formatSig(data.psd_variance_g2, 3)} g²`;
    $("#spectrumDominant").textContent = `${formatSig(data.effective_peak_frequency_hz ?? data.peak_frequency_hz, 4)} Hz`;
    $("#spectrumRate").textContent = `${formatSig((finite(data.sampling_rate_hz) || 0) / 1000, 4)} kHz`;
    $("#spectrumSource").textContent = data.metadata?.data_is_current ? "Live" : "Saved";
    const rows = $("#peakRows"); rows.replaceChildren();
    (data.top_peaks || []).slice(0, 5).forEach((peak, index) => {
      const row = document.createElement("tr");
      [String(index + 1), `${formatSig(peak.frequency_hz, 4)} Hz`, formatSig(peak.psd_g2_per_hz, 3), finite(peak.prominence_db) === null ? "-" : `${formatSig(peak.prominence_db, 3)} dB`].forEach((text, cellIndex) => { const cell = document.createElement("td"); if (cellIndex > 0) cell.className = "num"; cell.textContent = text; row.append(cell); });
      rows.append(row);
    });
    if (!rows.children.length) rows.innerHTML = "<tr><td>1</td><td>-</td><td>-</td><td>-</td></tr>";
  }

  function spectrumContext() {
    const data = state.spectrum || {};
    return {
      source: "spectrum",
      board: $("#spectrumBoard")?.value,
      axis: state.spectrumAxis,
      estimator: data.estimator,
      rms_g: data.psd_rms_g,
      variance_g2: data.psd_variance_g2,
      dominant_frequency_hz: data.effective_peak_frequency_hz ?? data.peak_frequency_hz,
      sampling_rate_hz: data.sampling_rate_hz,
      top_peaks: (data.top_peaks || []).slice(0, 5),
    };
  }

  function setupSpectrum() {
    setupSegments($("#spectrumAxis"), (value) => { state.spectrumAxis = value; });
    $("#spectrumBoard")?.addEventListener("change", () => { updateSpectrumBoardDot(); if (!state.spectrum) $("#spectrumStatus").textContent = `${boardLabel(selectedSensor()?.sensor_id)} selected · run a spectrum to analyze it`; });
    $("#runSpectrum")?.addEventListener("click", runSpectrum);
    $("#spectrumBands")?.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-min]"); if (!button) return;
      $("#spectrumBands").querySelectorAll("button").forEach((item) => item.classList.toggle("selected", item === button));
      state.spectrumRange = [Number(button.dataset.min), Number(button.dataset.max)]; drawSpectrum();
    });
    $("#askSpectrum")?.addEventListener("click", () => { if (!state.spectrum) return; storeContext(spectrumContext()); location.href = "/ask"; });
    window.addEventListener("resize", () => { if (state.spectrum) drawSpectrum(); });
  }

  async function loadReplays(preserveSelection = false) {
    const selectedId = preserveSelection ? state.replay?.window_id : null;
    try {
      const payload = await api("/api/vibro/replays?limit=50"); state.replays = payload.windows || [];
    } catch (error) { console.warn(error); state.replays = []; }
    state.replayIndex = selectedId ? state.replays.findIndex((item) => item.window_id === selectedId) : -1;
    renderReplayList();
    if (state.replays.length && state.replayIndex < 0) selectReplay(0);
  }

  const replayStateLabel = (value) => value === "data_invalid" ? "Data invalid" : (value === "normal" ? "Normal" : (value === "inconclusive" ? "Inconclusive" : "Anomaly"));
  const formatClockSeconds = (value) => { const date = new Date(value || ""); return Number.isNaN(date.getTime()) ? "-" : date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }); };
  const formatDay = (value) => { const date = new Date(value || ""); return Number.isNaN(date.getTime()) ? "" : date.toLocaleDateString([], { month: "short", day: "numeric" }); };

  function renderReplayList() {
    const list = $("#replayList"); list.replaceChildren();
    const count = $("#replayCount"); if (count) count.textContent = state.replays.length ? `${state.replays.length} saved` : "";
    if (!state.replays.length) { const empty = document.createElement("p"); empty.className = "empty-copy"; empty.style.padding = "12px 24px"; empty.textContent = "No saved checks yet. Anomalies, inconclusive checks, and invalid windows are saved here automatically."; list.append(empty); return; }
    state.replays.forEach((item, index) => {
      const button = document.createElement("button"); button.type = "button"; button.className = "replay-item"; button.classList.toggle("selected", index === state.replayIndex);
      const when = document.createElement("span"); when.className = "replay-when";
      const clock = document.createElement("time"); clock.dateTime = item.created_at_utc || ""; clock.textContent = formatClockSeconds(item.created_at_utc);
      const day = document.createElement("small"); day.textContent = formatDay(item.created_at_utc);
      when.append(clock, day);
      const chip = document.createElement("span"); chip.className = `state-chip ${item.state}`;
      chip.textContent = item.state === "data_invalid" ? "Data invalid" : (item.state === "normal" ? "Normal" : (item.state === "inconclusive" ? "Inconclusive" : "Anomaly"));
      const total = (item.affected_sensor_ids || []).length;
      const detail = document.createElement("span"); detail.className = "replay-count";
      detail.textContent = item.state === "data_invalid" ? "Signal quality" : (item.localization_status === "unresolved_global_anomaly" ? "Unlocalized" : (item.localization_status === "geometry_unavailable" ? `${total} candidate${total === 1 ? "" : "s"}` : (total === 5 ? "Network-wide" : (total ? `${total} target${total === 1 ? "" : "s"}` : "No targets"))));
      const meta = document.createElement("span"); meta.className = "replay-meta"; meta.append(chip, detail);
      button.append(when, meta); button.addEventListener("click", () => selectReplay(index)); list.append(button);
    });
  }

  async function selectReplay(index) {
    if (index < 0 || index >= state.replays.length) return;
    state.replayIndex = index; renderReplayList();
    try { state.replay = await api(`/api/vibro/replay?window_id=${encodeURIComponent(state.replays[index].window_id)}`); }
    catch (error) { state.replay = null; $("#replayExplanation").textContent = error.message; }
    renderReplay();
  }

  function renderReplay() {
    $("#replayPrevious").disabled = state.replayIndex <= 0; $("#replayNext").disabled = state.replayIndex < 0 || state.replayIndex >= state.replays.length - 1;
    if (!state.replay) return;
    const replay = state.replay; $("#replayTime").textContent = `${formatDate(replay.created_at_utc)} · ${replay.window_id}`;
    $("#replayState").textContent = replayStateLabel(replay.state);
    const card = $("#replayVerdict"); card.className = "verdict-card replay-verdict";
    card.classList.add(replay.state === "normal" ? "normal" : (replay.state === "inconclusive" ? "shared" : (replay.state === "data_invalid" ? "quality" : "anomaly")));
    setIcon($("#replayVerdictIcon"), replay.state === "normal" ? "check-circle" : (replay.state === "data_invalid" || replay.state === "inconclusive" ? "shield" : "warning-circle"));
    const affected = replay.affected_sensor_ids || []; const sharedPattern = affected.length === 5;
    $("#replayAffected").textContent = replay.state === "inconclusive"
      ? "Conflicting global and target decisions; no localized anomaly confirmed"
      : replay.localization_status === "unresolved_global_anomaly"
      ? "Network anomaly detected; target localization unavailable"
      : (replay.localization_status === "geometry_unavailable"
        ? `Model candidates: ${affected.map(boardLabel).join(", ")}; physical localization unavailable`
        : (sharedPattern ? "Network-wide result; all targets affected" : (affected.length ? `${affected.map(boardLabel).join(" and ")} affected` : "No affected targets")));
    const tbody = $("#replayBoards"); tbody.replaceChildren();
    const ribbon = [];
    (replay.boards || []).forEach((board, boardIndex) => {
      const row = document.createElement("tr");
      const color = BOARD_COLORS[boardIndex % BOARD_COLORS.length];
      const frequency = board.dominant_frequency_status === "legacy_wideband_only" ? "Not stored" : (finite(board.dominant_frequency_hz) === null ? "-" : `${formatSig(board.dominant_frequency_hz, 4)} Hz`);
      const candidate = replay.state === "inconclusive" || replay.localization_status === "geometry_unavailable";
      const kind = board.state === "data_invalid" ? "quality" : (board.state === "anomaly" ? (candidate || sharedPattern ? "shared" : "anomaly") : (board.state === "normal" ? "normal" : (board.state === "reference" ? "reference" : "unavailable")));
      const stateText = board.state === "data_invalid" ? "Data invalid" : (board.state === "anomaly" ? (candidate ? "Anomaly candidate" : (sharedPattern ? "Not localized" : classLabel(board.class))) : (board.state === "normal" ? "Normal" : (board.state === "reference" ? "Reference" : "Unavailable")));
      ribbon.push({ sensor_id: board.sensor_id, color, kind: kind === "reference" ? "normal" : kind, label: stateText });
      const labels = [boardLabel(board.sensor_id), formatSig(board.rms_g, 4), formatSig(board.peak_g, 4), frequency, stateText];
      labels.forEach((text, cellIndex) => {
        const cell = document.createElement("td");
        if (cellIndex === 0) {
          cell.className = "board-name-cell";
          const marker = document.createElement("span"); marker.className = "board-ident"; marker.style.setProperty("--board-color", color);
          const label = document.createElement("span"); label.textContent = text; cell.append(marker, label);
        } else if (cellIndex === 4) { cell.className = `board-state ${kind}`; cell.textContent = text; }
        else { cell.className = "num"; cell.textContent = text; }
        row.append(cell);
      }); tbody.append(row);
    });
    renderRibbon($("#replayRibbon"), ribbon);
    $("#replayExplanation").textContent = replay.explanation || "-";
  }

  function replayContext() {
    const replay = state.replay || {};
    const targets = (replay.boards || []).filter((board) => board.state !== "reference").map((board) => ({ sensor_id: board.sensor_id, affected: board.state === "anomaly", class: board.class, severity: board.severity, localization_candidate: board.state === "anomaly" && (replay.state === "inconclusive" || replay.localization_verified !== true) }));
    return { source: "replay", window_id: replay.window_id, created_at_utc: replay.created_at_utc, available: true, network_state: replay.state, state: replay.state, global_class: replay.global_class, global_target_consistent: replay.global_target_consistent, localization_status: replay.localization_status, localization_verified: replay.localization_verified, affected_sensor_ids: replay.affected_sensor_ids || [], targets, boards: replay.boards || [], explanation: replay.explanation || "", deployment: replay.deployment || {} };
  }

  function setupReplay() {
    $("#replayPrevious")?.addEventListener("click", () => selectReplay(state.replayIndex - 1));
    $("#replayNext")?.addEventListener("click", () => selectReplay(state.replayIndex + 1));
    $("#askReplay")?.addEventListener("click", () => { if (!state.replay) return; storeContext(replayContext()); location.href = "/ask"; });
  }

  function storeContext(context) { if (context) sessionStorage.setItem("vibroagentGemmaContext", JSON.stringify(context)); }
  function loadStoredContext() {
    try { const value = JSON.parse(sessionStorage.getItem("vibroagentGemmaContext") || "null"); sessionStorage.removeItem("vibroagentGemmaContext"); return value; }
    catch { return null; }
  }

  const THREAD_KEY = "vibroagentGemmaThread";
  const STARTERS = {
    latest_gemma_check: ["What did the latest check find?", "Which boards need attention?", "Is this localized or network-wide?", "How fresh is this check?"],
    spectrum: ["What is the dominant frequency?", "Summarize the top peaks.", "Does this spectrum look unusual?", "Which band carries the most energy?"],
    replay: ["Summarize this saved check.", "Which targets were affected?", "Was the localization verified?", "What should I review next?"],
  };
  const INLINE_MARKUP = /\*\*([^*\n]+)\*\*|`([^`\n]+)`|(^|[^\w*])\*([^*\n]+)\*(?![\w*])|(^|[^\w])_([^_\n]+)_(?!\w)/g;
  const chat = { messages: [], pending: false, entries: new WeakMap(), welcomeKey: null };
  const welcomeKey = () => `${state.context?.source || "latest_gemma_check"}|${state.context?.available === false}`;
  const reducedMotion = () => Boolean(window.matchMedia?.("(prefers-reduced-motion: reduce)").matches);
  const formatClock = (value) => {
    const date = new Date(value || Date.now());
    return Number.isNaN(date.getTime()) ? "" : date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  };

  function stateLabel(value) {
    const names = { normal: "Normal", anomaly: "Anomaly", inconclusive: "Inconclusive", data_invalid: "Data invalid", shared_anomaly: "Network-level advisory", unavailable: "Unavailable" };
    const key = String(value || "");
    return names[key] || (key ? key.charAt(0).toUpperCase() + key.slice(1) : "-");
  }

  function describeContext(context) {
    if (context?.source === "spectrum") {
      const axis = context.axis && context.axis !== "norm" ? `${String(context.axis).toUpperCase()}-axis ` : "";
      return `the ${axis}spectrum for ${boardLabel(context.board)}`;
    }
    if (context?.source === "replay") return `the saved check from ${formatDate(context.created_at_utc)}`;
    return "the latest Agent check";
  }

  function renderContext() {
    const context = state.context || { source: "latest_gemma_check", available: false, targets: [] };
    const sourceNames = { latest_gemma_check: "Latest Agent check", spectrum: "Spectrum", replay: "Saved replay" };
    $("#contextSource").textContent = sourceNames[context.source] || "Agent context";
    const contextState = $("#contextState"); const contextStateContainer = contextState.parentElement;
    const stateName = context.available === false ? "unavailable" : String(context.network_state || context.state || "");
    contextState.textContent = stateLabel(stateName);
    contextStateContainer.className = stateName === "normal" ? "normal" : (["anomaly", "shared_anomaly"].includes(stateName) ? "anomaly" : "");
    setIcon($("#contextStateIcon"), stateName === "normal" ? "check-circle" : (["anomaly", "shared_anomaly"].includes(stateName) ? "warning-circle" : (stateName === "data_invalid" ? "shield" : "clock")));
    const boardList = $("#contextBoards"); boardList.replaceChildren();
    const targets = context.targets || [];
    const sharedPattern = isSharedTargetPattern(targets);
    const rows = [{ sensor_id: state.baselineId, state: "Reference" }, ...targets.map((target) => ({ sensor_id: target.sensor_id, state: target.class === "data_invalid" ? "Data invalid" : (target.localization_candidate ? "Anomaly candidate" : (sharedPattern ? "Not localized" : (target.affected ? classLabel(target.class) : "Normal"))) }))];
    const rowKind = (row) => !['Reference', 'Normal'].includes(row.state) ? (row.state === "Not localized" ? "shared" : (row.state === "Data invalid" ? "quality" : "anomaly")) : "normal";
    rows.forEach((row, index) => {
      const element = document.createElement("div"); element.className = "state-row"; element.style.setProperty("--board-color", BOARD_COLORS[index % BOARD_COLORS.length]);
      const dot = document.createElement("span"); dot.className = "state-dot"; const name = document.createElement("span"); name.textContent = boardLabel(row.sensor_id); const value = document.createElement("span"); value.className = `state-value ${rowKind(row)}`; value.textContent = row.state;
      element.append(dot, name, value); boardList.append(element);
    });
    renderRibbon($("#contextRibbon"), rows.length > 1 ? rows.map((row, index) => ({ sensor_id: row.sensor_id, color: BOARD_COLORS[index % BOARD_COLORS.length], kind: rowKind(row), label: row.state })) : []);
    $("#contextTokens").textContent = `${context.soft_tokens_valid || "-"} continuous tokens`;
    $("#contextUpdated").textContent = `Updated ${formatAge(context.updated_age_s)}`;
    renderThreadStatus();
    renderComposerContext();
    if (!chat.messages.length && !chat.pending && welcomeKey() !== chat.welcomeKey) renderThread();
  }

  function renderThreadStatus() {
    const status = $("#threadStatus"); const text = $("#threadStatusText");
    if (!status || !text) return;
    const context = state.context || {};
    status.className = "thread-status";
    if (chat.pending) { status.classList.add("busy"); text.textContent = "Agent is thinking…"; return; }
    if (context.available === false) {
      status.classList.add("limited");
      text.textContent = context.stale
        ? "Latest Agent check is stale · answers are limited until a fresh check completes"
        : "Latest Agent check unavailable · answers are limited";
      return;
    }
    status.classList.add("online");
    let detail = "";
    if (context.source === "spectrum") detail = ` · ${formatNumber(context.dominant_frequency_hz, 2)} Hz dominant`;
    else if (context.source === "replay") detail = ` · ${stateLabel(context.state || context.network_state)}`;
    else detail = ` · ${stateLabel(context.network_state)} · updated ${formatAge(context.updated_age_s).toLowerCase()}`;
    text.textContent = `Online · answering from ${describeContext(context)}${detail}`;
  }

  function renderComposerContext() {
    const context = state.context || {};
    const input = $("#chatInput"); const hint = $("#composerContext");
    const placeholders = { spectrum: "Ask about this spectrum…", replay: "Ask about this saved check…" };
    if (input) input.placeholder = placeholders[context.source] || "Ask about the latest Agent check…";
    if (hint) hint.textContent = context.available === false ? "Live evidence unavailable · the Agent will say what it cannot answer" : `Answers come from ${describeContext(context)}`;
  }

  function appendInline(parent, text) {
    INLINE_MARKUP.lastIndex = 0;
    let last = 0; let match;
    while ((match = INLINE_MARKUP.exec(text)) !== null) {
      const [token, bold, code, emPrefix, em, underPrefix, under] = match;
      if (match.index > last) parent.append(text.slice(last, match.index));
      if (emPrefix) parent.append(emPrefix); else if (underPrefix) parent.append(underPrefix);
      const element = document.createElement(bold !== undefined ? "strong" : (code !== undefined ? "code" : "em"));
      element.textContent = bold ?? code ?? em ?? under;
      parent.append(element);
      last = match.index + token.length;
    }
    if (last < text.length) parent.append(text.slice(last));
  }

  function renderRichText(container, content) {
    container.replaceChildren();
    const text = String(content || "").replace(/\r\n?/g, "\n").trim();
    const BULLET = /^[-*•]\s+/; const NUMBERED = /^\d+[.)]\s+/; const HEADING = /^#{1,6}\s+/;
    text.split(/\n{2,}/).forEach((block) => {
      let paragraph = null; let list = null;
      block.split("\n").forEach((raw) => {
        const line = raw.trim();
        if (!line) return;
        const kind = BULLET.test(line) ? "ul" : (NUMBERED.test(line) ? "ol" : null);
        if (kind) {
          paragraph = null;
          if (!list || list.tagName.toLowerCase() !== kind) { list = document.createElement(kind); container.append(list); }
          const item = document.createElement("li"); appendInline(item, line.replace(kind === "ul" ? BULLET : NUMBERED, "")); list.append(item);
          return;
        }
        list = null;
        if (HEADING.test(line)) {
          const heading = document.createElement("p"); heading.className = "rich-heading"; appendInline(heading, line.replace(HEADING, "")); container.append(heading); paragraph = null;
          return;
        }
        if (!paragraph) { paragraph = document.createElement("p"); container.append(paragraph); } else paragraph.append(document.createElement("br"));
        appendInline(paragraph, line);
      });
    });
    if (!container.childNodes.length) container.textContent = text;
  }

  function saveThread() {
    try { sessionStorage.setItem(THREAD_KEY, JSON.stringify({ messages: chat.messages.slice(-80), history: state.history.slice(-6), context: describeContext(state.context) })); } catch { /* storage unavailable */ }
  }
  function loadThread() {
    try {
      const stored = JSON.parse(sessionStorage.getItem(THREAD_KEY) || "null");
      chat.messages = Array.isArray(stored?.messages) ? stored.messages.filter((entry) => entry && ["user", "assistant", "system"].includes(entry.role) && typeof entry.content === "string") : [];
      state.history = Array.isArray(stored?.history) ? stored.history.filter((item) => item && ["user", "assistant"].includes(item.role) && typeof item.content === "string") : [];
      return typeof stored?.context === "string" ? stored.context : null;
    } catch { chat.messages = []; state.history = []; return null; }
  }

  function welcomeElement() {
    const context = state.context || {};
    const wrap = document.createElement("div"); wrap.className = "thread-welcome";
    const mark = document.createElement("span"); mark.className = "welcome-mark"; mark.setAttribute("aria-hidden", "true");
    const title = document.createElement("h2"); title.textContent = "Hi, I'm the VibroAgent assistant.";
    const copy = document.createElement("p");
    copy.textContent = context.available === false
      ? "The latest Agent check is not available right now, so I can only explain what is missing. Open the live monitor to see when a fresh check completes."
      : `Ask me about ${describeContext(context)}. I answer only from the evidence in the current context and never guess measurements, faults, or causes.`;
    const chips = document.createElement("div"); chips.className = "starter-chips";
    (STARTERS[context.source] || STARTERS.latest_gemma_check).forEach((starter) => {
      const chip = document.createElement("button"); chip.type = "button"; chip.className = "starter-chip"; chip.textContent = starter; chips.append(chip);
    });
    wrap.append(mark, title, copy, chips);
    return wrap;
  }

  function noticeElement(entry) {
    const notice = document.createElement("div"); notice.className = "thread-notice"; notice.setAttribute("role", "status");
    notice.append(makeIcon(entry.icon || "file-text"), document.createTextNode(entry.content));
    chat.entries.set(notice, entry);
    return notice;
  }

  function messageElement(entry, previous) {
    const article = document.createElement("article");
    article.className = `message ${entry.role}${entry.kind ? ` ${entry.kind}` : ""}`;
    const grouped = Boolean(previous && previous.role === entry.role && !previous.kind && !entry.kind && Math.abs(new Date(entry.at) - new Date(previous.at)) < 3 * 60 * 1000);
    if (grouped) article.classList.add("grouped");
    const avatar = document.createElement("span"); avatar.className = "message-avatar"; avatar.setAttribute("aria-hidden", "true");
    if (entry.role === "user") avatar.append(makeIcon("user"));
    const body = document.createElement("div"); body.className = "message-body";
    const meta = document.createElement("div"); meta.className = "message-meta";
    const author = document.createElement("span"); author.textContent = entry.role === "user" ? "You" : "Agent";
    const time = document.createElement("time"); time.dateTime = entry.at || ""; time.textContent = formatClock(entry.at);
    meta.append(author, time);
    const bubble = document.createElement("div"); bubble.className = "message-bubble";
    if (entry.kind === "error") {
      bubble.append(document.createTextNode(entry.content));
      if (entry.detail) { const detail = document.createElement("span"); detail.className = "error-detail"; detail.textContent = entry.detail; bubble.append(detail); }
    } else if (entry.role === "assistant") renderRichText(bubble, entry.content);
    else bubble.textContent = entry.content;
    body.append(meta, bubble);
    if (entry.kind === "error") {
      const actions = document.createElement("div"); actions.className = "message-actions";
      const retry = document.createElement("button"); retry.type = "button"; retry.className = "retry-button";
      retry.append(makeIcon("replay"), document.createTextNode("Try again"));
      retry.addEventListener("click", () => retryAfter(entry));
      actions.append(retry); body.append(actions);
    }
    article.append(avatar, body);
    chat.entries.set(article, entry);
    return article;
  }

  function renderEntry(entry, previous) { return entry.role === "system" ? noticeElement(entry) : messageElement(entry, previous); }

  function renderThread() {
    const container = $("#messages"); if (!container) return;
    container.replaceChildren();
    chat.welcomeKey = welcomeKey();
    if (!chat.messages.length) { container.append(welcomeElement()); return; }
    let previous = null;
    chat.messages.forEach((entry) => { container.append(renderEntry(entry, previous)); previous = entry.role === "system" ? null : entry; });
  }

  function scrollThread(force, instant = false) {
    const thread = $("#thread"); if (!thread) return;
    const nearBottom = thread.scrollHeight - thread.scrollTop - thread.clientHeight < 160;
    if (force || nearBottom) thread.scrollTo({ top: thread.scrollHeight, behavior: instant || reducedMotion() ? "auto" : "smooth" });
  }

  function pushMessage(entry) {
    const container = $("#messages"); if (!container) return null;
    if (!chat.messages.length) container.replaceChildren();
    const last = chat.messages[chat.messages.length - 1];
    const element = renderEntry(entry, last && last.role !== "system" ? last : null);
    chat.messages.push(entry);
    element.classList.add("enter");
    container.append(element);
    saveThread(); scrollThread(true);
    return element;
  }

  function removeEntry(entry) {
    const index = chat.messages.indexOf(entry);
    if (index >= 0) chat.messages.splice(index, 1);
    $$("#messages > *").forEach((element) => { if (chat.entries.get(element) === entry) element.remove(); });
    saveThread();
  }

  function showTyping() {
    const article = document.createElement("article"); article.className = "message assistant typing enter"; article.setAttribute("aria-label", "Agent is typing");
    const avatar = document.createElement("span"); avatar.className = "message-avatar"; avatar.setAttribute("aria-hidden", "true");
    const body = document.createElement("div"); body.className = "message-body";
    const bubble = document.createElement("div"); bubble.className = "message-bubble";
    const dots = document.createElement("span"); dots.className = "typing-dots";
    for (let index = 0; index < 3; index += 1) dots.append(document.createElement("i"));
    bubble.append(dots); body.append(bubble); article.append(avatar, body);
    $("#messages")?.append(article); scrollThread(true);
    return article;
  }

  function describeError(error) {
    const raw = String(error?.message || error || "Request failed");
    if (/busy/i.test(raw)) return { content: "The Agent is busy with another request. Give it a moment and try again.", detail: null };
    if (/refused|urlopen|timed? ?out|unreachable|502|504/i.test(raw)) return { content: "The Agent model did not respond. Check that the model server is running, then try again.", detail: raw };
    if (/too long/i.test(raw)) return { content: "That message is too long for the Agent. Please shorten it and try again.", detail: null };
    return { content: "The Agent could not answer this message.", detail: raw };
  }

  function lastUserQuestion() {
    for (let index = chat.messages.length - 1; index >= 0; index -= 1) if (chat.messages[index].role === "user") return chat.messages[index].content;
    return "";
  }

  async function askAgent(question) {
    if (chat.pending) return;
    chat.pending = true; setComposerBusy(true); renderThreadStatus();
    const typing = showTyping();
    try {
      const payload = await api("/api/vibro/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message: question, context: state.context, history: state.history }) });
      typing.remove();
      const answer = String(payload.answer || "").trim();
      if (payload.context?.available === false) {
        pushMessage({ role: "system", icon: "clock", content: answer || "The latest Agent check is unavailable, so the Agent cannot answer from live evidence yet.", at: new Date().toISOString() });
      } else {
        pushMessage({ role: "assistant", content: answer || "The Agent returned an empty answer.", at: new Date().toISOString() });
        state.history = [...state.history, { role: "user", content: question }, { role: "assistant", content: answer }].slice(-6);
        saveThread();
      }
    } catch (error) {
      typing.remove(); console.warn(error);
      pushMessage({ role: "assistant", kind: "error", ...describeError(error), question, at: new Date().toISOString() });
    } finally {
      chat.pending = false; setComposerBusy(false); renderThreadStatus(); $("#chatInput")?.focus();
    }
  }

  function submitQuestion(value) {
    const question = String(value || "").trim();
    if (!question || chat.pending) return;
    const input = $("#chatInput"); if (input) { input.value = ""; autosizeComposer(); }
    pushMessage({ role: "user", content: question, at: new Date().toISOString() });
    askAgent(question);
  }

  function retryAfter(entry) {
    const question = entry.question || lastUserQuestion();
    removeEntry(entry);
    if (question) askAgent(question);
  }

  function autosizeComposer() {
    const input = $("#chatInput"); if (!input) return;
    input.style.height = "auto"; input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
  }
  function updateSendState() {
    const input = $("#chatInput"); const button = $("#sendChat"); if (!input || !button) return;
    button.disabled = chat.pending || !input.value.trim();
  }
  function setComposerBusy(busy) { $("#chatForm")?.classList.toggle("busy", busy); updateSendState(); }

  function newConversation() {
    chat.messages = []; state.history = [];
    try { sessionStorage.removeItem(THREAD_KEY); } catch { /* ignore */ }
    renderThread(); $("#chatInput")?.focus();
  }

  function setupAsk() {
    const stored = loadStoredContext();
    const previousContext = loadThread();
    if (stored) {
      state.context = stored;
      if (chat.messages.length && describeContext(stored) !== previousContext) chat.messages.push({ role: "system", icon: stored.source === "spectrum" ? "waveform" : (stored.source === "replay" ? "replay" : "file-text"), content: `Now discussing ${describeContext(stored)}.`, at: new Date().toISOString() });
      saveThread();
    }
    renderContext(); renderThread(); scrollThread(true, true);
    const input = $("#chatInput"); const form = $("#chatForm");
    form?.addEventListener("submit", (event) => { event.preventDefault(); submitQuestion(input?.value); });
    input?.addEventListener("input", () => { autosizeComposer(); updateSendState(); });
    input?.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" || event.shiftKey || event.isComposing) return;
      event.preventDefault(); submitQuestion(input.value);
    });
    $("#messages")?.addEventListener("click", (event) => { const chip = event.target.closest(".starter-chip"); if (chip) submitQuestion(chip.textContent); });
    $("#newChat")?.addEventListener("click", newConversation);
    autosizeComposer(); updateSendState(); input?.focus();
  }

  async function init() {
    configureView(); setupMonitor(); setupSpectrum(); setupReplay();
    await loadSensors();
    await loadSyntheticInjection();
    await fetchMonitor();
    if (state.view === "monitor") { await fetchWaveforms(); setInterval(fetchWaveforms, 2000); setInterval(fetchMonitor, 5000); }
    if (state.view === "replay") { await loadReplays(); setInterval(() => loadReplays(true), 10000); }
    if (state.view === "ask") { setupAsk(); setInterval(fetchMonitor, 5000); }
  }

  init().catch((error) => { console.error(error); setLiveState(false, "App unavailable"); });
})();
