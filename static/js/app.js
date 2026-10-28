const state = { page: "overview", zones: [], predictions: [], graph: { edges: [] }, scenarios: [], plan: null, selectedZone: null, mapMode: "risk", analytics: null, dataset: null, models: {}, activeModel: "spatiotemporal_gnn", computedAt: null };
const researchTitle = "An Uncertainty-Aware Spatio-Temporal Graph Reinforcement Learning Framework for Climate-Resilient Green Infrastructure Planning in Smart Cities";
const pages = {
  overview: ["Overview", "RESEARCH OVERVIEW", "Climate risk, model uncertainty and adaptation planning in one study area."],
  map: ["Urban risk map", "SPATIAL GRAPH", "Place-level climate exposure positioned from geographic coordinates."],
  flood: ["Flood risk", "PLUVIAL FLOOD EXPOSURE", "Interpretable flood-risk estimates from rainfall, drainage and surface conditions."],
  heat: ["Heat risk", "URBAN HEAT EXPOSURE", "Heat-risk estimates shaped by temperature, density and green coverage."],
  uncertainty: ["Uncertainty analysis", "MONTE CARLO DROPOUT", "Prediction spread is shown separately from estimated hazard magnitude."],
  recommendations: ["Green infrastructure", "INTERVENTION PLAN", "Zone-level actions, implementation assumptions and estimated co-benefits."],
  optimization: ["RL optimization", "POLICY DESIGN", "Train a PPO policy under a fixed budget and configurable multi-objective priorities."],
  scenarios: ["Climate scenarios", "STRESS TESTS", "Compare simulated baseline, rainfall, heatwave and compound conditions."],
  explain: ["Explainable AI", "MODEL ATTRIBUTION", "Inspect integrated-gradient feature contributions and local recommendation context."],
  comparison: ["Before / after", "PLAN OUTCOMES", "Compare the unmitigated risk state with estimated plan-level changes."],
  analytics: ["Analytics", "EXPERIMENT OUTPUTS", "Fit diagnostics, strategy baselines, uncertainty and ablation comparisons."],
  system: ["System information", "METHOD & PROVENANCE", "Prototype workflow, model controls, data caveats and local API surface."],
  "result-graphs": ["Result graphs", "PUBLICATION FIGURES", "Generated from the active dataset; 1200-DPI PNG and vector PDF exports."]
};
const $ = (selector, root = document) => root.querySelector(selector);
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
const pct = (value, digits = 0) => `${(Number(value || 0) * 100).toFixed(digits)}%`;
const money = (value) => new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(Number(value || 0));
const fixed = (value, digits = 1) => Number(value || 0).toFixed(digits);
const api = async (url, options = {}) => {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
};
function updateFreshness(timestamp = new Date().toISOString()) {
  state.computedAt = timestamp;
  const element = $("#freshness");
  if (!element) return;
  element.textContent = `Computed ${new Date(timestamp).toLocaleTimeString()}`;
  element.title = "Values are recomputed from the active dataset every 30 seconds; this is not a live sensor feed.";
}
function updateDatasetBadge(dataset) {
  const simulated = dataset?.source_type === "seeded_simulation";
  const label = $(".sample-label");
  if (label) label.lastChild.textContent = simulated ? " DEMONSTRATION DATA" : " USER DATA · VERIFY PROVENANCE";
  if ($("#footerStatus")) $("#footerStatus").textContent = `${dataset?.row_count || state.zones.length} ${simulated ? "demonstration" : "uploaded"} places`;
}
function mountModelSelector() {
  const learningRate = $("#learningRate");
  if (learningRate) learningRate.step = "0.0001";
  const grid = $("#trainingForm .control-grid");
  if (!grid || $("#modelType")) return;
  const field = document.createElement("div");
  field.className = "field";
  field.innerHTML = `<label for="modelType">Prediction model</label><select id="modelType"></select>`;
  const select = $("#modelType", field);
  for (const [key, name] of Object.entries(state.models)) {
    const option = document.createElement("option");
    option.value = key;
    option.textContent = name;
    option.selected = key === state.activeModel;
    select.append(option);
  }
  grid.prepend(field);
  const note = $("#trainingForm .control-footer .small-note");
  if (note) note.textContent = `Seed ${state.models.seed ?? 42} · ${state.device || "CPU"} · proxy-label fit`;
}

function toast(message, error = false) {
  const el = $("#notice"); el.textContent = message; el.className = `notice${error ? " error" : ""}`; el.style.display = "block";
  window.clearTimeout(toast.timer); toast.timer = window.setTimeout(() => { el.style.display = "none"; }, 5400);
}
function riskColor(value) { return value >= .68 ? "#79394b" : value >= .48 ? "#50386f" : value >= .3 ? "#176d6b" : "#205541"; }
function deepPaletteMarkup(markup) {
  return markup.replaceAll("#d9ff54", "#24563f").replaceAll("var(--yellow)", "var(--accent)");
}
function uncertaintyColor(value) {
  const low = [77, 154, 112], high = [182, 160, 255], ratio = Math.max(0, Math.min(1, Number(value || 0) / .3));
  return `#${low.map((channel, index) => Math.round(channel + (high[index] - channel) * ratio).toString(16).padStart(2, "0")).join("")}`;
}
function riskClass(band) { return String(band || "").toLowerCase(); }
function metricCard(label, value, note, color = "var(--green)") {
  return `<article class="kpi" style="--kpi-color:${color}"><div class="kpi-label">${label}</div><div class="kpi-value">${value}</div><div class="kpi-note">${note}</div></article>`;
}
function header(title, eyebrow, subtitle, actions = "") {
  return `<div class="page-heading"><div><div class="eyebrow">${eyebrow}</div><h1>${title}</h1><p class="subtitle">${subtitle}</p></div><div class="heading-actions">${actions}</div></div>`;
}
function panel(title, subtitle, body, actions = "") {
  return `<section class="panel"><div class="panel-head"><div><div class="panel-title">${title}</div>${subtitle ? `<div class="panel-subtitle">${subtitle}</div>` : ""}</div>${actions}</div><div class="panel-body">${body}</div></section>`;
}
function avg(key) { return state.predictions.length ? state.predictions.reduce((sum, row) => sum + Number(row[key] || 0), 0) / state.predictions.length : 0; }
function getZone(zoneId) { return state.zones.find((zone) => zone.zone_id === zoneId); }
function getPrediction(zoneId) { return state.predictions.find((row) => row.zone_id === zoneId); }

function mapMarkup() {
  if (!state.zones.length) return `<div class="empty-state">Load a valid zone dataset to draw the spatial graph.</div>`;
  const longitudeScale = Math.cos(state.zones.reduce((sum, zone) => sum + Number(zone.latitude), 0) / state.zones.length * Math.PI / 180);
  const projected = state.zones.map((zone) => ({ x: Number(zone.longitude) * longitudeScale, y: Number(zone.latitude) }));
  const minX = Math.min(...projected.map((point) => point.x)), maxX = Math.max(...projected.map((point) => point.x));
  const minY = Math.min(...projected.map((point) => point.y)), maxY = Math.max(...projected.map((point) => point.y));
  const scale = Math.min(660 / (maxX - minX || 1), 250 / (maxY - minY || 1));
  const centerX = (minX + maxX) / 2, centerY = (minY + maxY) / 2;
  const point = (zone) => ({ x: 390 + (Number(zone.longitude) * longitudeScale - centerX) * scale,
    y: 160 - (Number(zone.latitude) - centerY) * scale });
  const zones = new Map(state.zones.map((zone) => [zone.zone_id, zone]));
  const edges = (state.graph.edges || []).map(([a, b]) => {
    const one = zones.get(a), two = zones.get(b); if (!one || !two) return "";
    const p1 = point(one), p2 = point(two); return `<line class="map-edge" x1="${p1.x}" y1="${p1.y}" x2="${p2.x}" y2="${p2.y}"/>`;
  }).join("");
  const nodes = state.zones.map((zone) => {
    const p = point(zone), prediction = getPrediction(zone.zone_id) || {};
    const riskKey = state.page === "flood" ? "flood_risk" : state.page === "heat" ? "heat_risk" : "vulnerability";
    const value = state.mapMode === "uncertainty" ? prediction.uncertainty : prediction[riskKey];
    const color = state.mapMode === "uncertainty" ? uncertaintyColor(value) : riskColor(value);
    const action = state.plan?.interventions?.find((item) => item.zone_id === zone.zone_id);
    const markerColors = { "Urban trees": "#205541", "Green roof": "#50386f", "Rain garden": "#176d6b", "Permeable pavement": "#255c79", "Urban wetland": "#79394b" };
    const markerGlyphs = { "Urban trees": "T", "Green roof": "R", "Rain garden": "G", "Permeable pavement": "P", "Urban wetland": "W" };
    const selected = state.selectedZone === zone.zone_id ? 10 : 7;
    const metricLabel = state.mapMode === "uncertainty" ? "uncertainty" : riskKey.replace("_risk", " risk");
    return `<g class="map-zone" data-zone="${esc(zone.zone_id)}" tabindex="0" role="button" aria-label="${esc(zone.name)}, ${metricLabel} ${pct(value)}">
      <circle class="map-node" cx="${p.x}" cy="${p.y}" r="${selected}" fill="${color}"/>
      ${action ? `<circle cx="${p.x}" cy="${p.y}" r="12" fill="none" stroke="#176d6b" stroke-width="1.5" stroke-dasharray="3 3"/><circle cx="${p.x + 8}" cy="${p.y - 9}" r="7" fill="${markerColors[action.action] || "#176d6b"}"/><text x="${p.x + 8}" y="${p.y - 6}" text-anchor="middle" fill="#fff" font-size="10" font-weight="700">${markerGlyphs[action.action] || "G"}</text>` : ""}
      <text class="map-label" x="${p.x + 12}" y="${p.y + 3}">${esc(zone.name)}</text>
      <title>${esc(zone.name)} · ${metricLabel} ${pct(value)}${action ? ` · ${esc(action.action)}` : ""}</title></g>`;
  }).join("");
  const studyArea = state.dataset?.name?.includes("Chennai") ? "CHENNAI, INDIA" : "ACTIVE STUDY AREA";
  const coordinateSource = state.dataset?.source_type === "seeded_simulation"
    ? "LOCALITY COORDINATES © OPENSTREETMAP CONTRIBUTORS · ENVIRONMENTAL ATTRIBUTES ARE DEMONSTRATION INPUTS"
    : "COORDINATES FROM ACTIVE DATASET · VERIFY SOURCE PROVENANCE";
  return `<div class="map-frame"><svg viewBox="0 0 780 320" role="img" aria-label="Georeferenced locality risk graph for ${studyArea}">
    ${edges}${nodes}</svg><div class="map-caption">${studyArea} · GEOGRAPHIC ASPECT PRESERVED · ${coordinateSource}</div></div>`;
}
function mapPanel() {
  const actions = `<div class="segmented"><button data-map-mode="risk" class="${state.mapMode === "risk" ? "selected" : ""}">Risk</button><button data-map-mode="uncertainty" class="${state.mapMode === "uncertainty" ? "selected" : ""}">Uncertainty</button></div>`;
  const legend = state.mapMode === "uncertainty" ? `<span style="--legend-color:#205541">Lower spread</span><span style="--legend-color:#50386f">Higher spread</span>` : `<span style="--legend-color:#205541">Lower</span><span style="--legend-color:#176d6b">Moderate</span><span style="--legend-color:#50386f">Elevated</span><span style="--legend-color:#79394b">High</span>`;
  const actionLegend = state.plan?.interventions?.length ? `<div class="legend" style="margin-top:10px"><span style="--legend-color:#205541">T · Trees</span><span style="--legend-color:#50386f">R · Roof</span><span style="--legend-color:#176d6b">G · Rain garden</span><span style="--legend-color:#255c79">P · Permeable</span><span style="--legend-color:#79394b">W · Wetland</span></div>` : "";
  return `<section class="panel"><div class="panel-head"><div><div class="panel-title">Zone graph</div><div class="panel-subtitle">3-nearest-neighbor · normalized adjacency</div></div><div class="map-toolbar">${actions}</div></div>${mapMarkup()}<div class="panel-body"><div class="legend">${legend}</div>${actionLegend}<div id="zoneDetail"></div></div></section>`;
}
function sparkBars(items, key, options = {}) {
  if (!items?.length) return `<div class="empty-state">No experiment outputs yet.</div>`;
  const width = 620, height = options.height || 205, margin = { top: 18, right: 13, bottom: 45, left: 36 };
  const areaW = width - margin.left - margin.right, areaH = height - margin.top - margin.bottom;
  const values = items.map((item) => Number(item[key] || 0)); const max = Math.max(...values, .01) * 1.15;
  const step = areaW / items.length; const barWidth = Math.min(35, step * .54);
  const lines = [0, .5, 1].map((ratio) => {
    const y = margin.top + areaH * ratio;
    return `<line class="chart-grid-line" x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}"/><text class="chart-tick-value" x="2" y="${y + 3}">${fixed(max * (1 - ratio), 1)}</text>`;
  }).join("");
  const bars = items.map((item, index) => {
    const value = values[index], h = value / max * areaH, x = margin.left + index * step + (step - barWidth) / 2;
    const label = options.label ? options.label(item) : item.name || item.scenario || item.zone_id || "";
    const fill = options.color ? options.color(item) : "#7fd0a2";
    return `<g><rect x="${x}" y="${margin.top + areaH - h}" width="${barWidth}" height="${Math.max(h, 1)}" rx="2" fill="${fill}"/><text class="chart-value" text-anchor="middle" x="${x + barWidth / 2}" y="${Math.max(11, margin.top + areaH - h - 5)}">${options.value ? options.value(value) : fixed(value, 2)}</text><text class="chart-label" text-anchor="middle" x="${x + barWidth / 2}" y="${height - 19}">${esc(label).slice(0, 11)}</text></g>`;
  }).join("");
  return `<div class="chart-wrap"><svg viewBox="0 0 ${width} ${height}" role="img">${lines}${bars}</svg></div>`;
}
function uncertaintyHistogram() {
  const edges = [0, .05, .1, .15, .2, .25, .3];
  const bins = edges.slice(0, -1).map((start, index) => ({
    name: `${Math.round(start * 100)}-${Math.round(edges[index + 1] * 100)}%`,
    count: state.predictions.filter((row) => row.uncertainty >= start && row.uncertainty < edges[index + 1]).length
  }));
  return sparkBars(bins, "count", { label: (row) => row.name, value: (value) => String(value), color: () => "#b6a0ff" });
}
function costResilienceScatter(comparisons) {
  const points = Object.entries(comparisons || {}).map(([name, row]) => ({ name, cost: row.total_cost, resilience: row.resilience_after }));
  if (!points.length) return `<div class="empty-state">Run an optimization to compare strategies.</div>`;
  const width = 560, height = 230, left = 58, right = 20, top = 20, bottom = 42;
  const maxCost = Math.max(...points.map((point) => point.cost), 1) * 1.1;
  const minScore = Math.min(...points.map((point) => point.resilience)) - .5;
  const maxScore = Math.max(...points.map((point) => point.resilience)) + .5;
  const y = (value) => top + (maxScore - value) / (maxScore - minScore || 1) * (height - top - bottom);
  const colors = ["#205541", "#176d6b", "#50386f"];
  const dots = points.map((point, index) => {
    const x = left + point.cost / maxCost * (width - left - right);
    return `<g><circle cx="${x}" cy="${y(point.resilience)}" r="6" fill="${colors[index % colors.length]}"/><text class="chart-label" x="${x + 9}" y="${y(point.resilience) + 3}">${esc(point.name)}</text><title>${esc(point.name)} · lifecycle ${money(point.cost)} · resilience ${fixed(point.resilience, 1)}</title></g>`;
  }).join("");
  return `<div class="chart-wrap"><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Lifecycle cost versus estimated resilience"><line class="chart-grid-line" x1="${left}" y1="${height-bottom}" x2="${width-right}" y2="${height-bottom}"/><line class="chart-grid-line" x1="${left}" y1="${top}" x2="${left}" y2="${height-bottom}"/><text class="chart-label" x="${left}" y="${height-10}">Lifecycle cost →</text><text class="chart-label" x="4" y="${top+4}">Resilience ↑</text>${dots}</svg></div>`;
}
function additionalCharts() {
  if (state.page === "analytics" && state.analytics) {
    const metrics = state.analytics.model;
    const fit = sparkBars([{ name: "MAE", value: metrics.mae }, { name: "RMSE", value: metrics.rmse }, { name: "R²", value: metrics.r2_proxy_fit }], "value", { label: (row) => row.name, color: (row) => row.name === "R²" ? "#b6a0ff" : "#7fd0a2" });
    return `<div class="layout-equal">${panel("Model fit diagnostics", "IN-SAMPLE GENERATED PROXY LABELS", fit)}${panel("Uncertainty distribution", "ZONE-LEVEL MC-DROPOUT SPREAD", uncertaintyHistogram())}</div>`;
  }
  if (state.page === "optimization" && state.plan) {
    return `<div style="margin-top:10px">${panel("Cost versus resilience", "LIFECYCLE COST · STRATEGY COMPARISON", costResilienceScatter(state.plan.comparisons))}</div>`;
  }
  if (state.page === "comparison" && state.plan) {
    const plan = state.plan;
    const risk = [{ name: "Flood before", value: plan.flood_before }, { name: "Flood after", value: plan.flood_after }, { name: "Heat before", value: plan.heat_before }, { name: "Heat after", value: plan.heat_after }];
    return `<div class="layout-equal">${panel("Before and after risk", "ESTIMATED ZONE-AVERAGE RISK", sparkBars(risk, "value", { label: (row) => row.name, color: (row) => row.name.endsWith("after") ? "#205541" : "#79394b" }))}${panel("Carbon reduction by zone", "ASSUMPTION-BASED CO-BENEFIT", sparkBars(plan.interventions, "carbon_tonnes", { label: (row) => row.zone_id, color: () => "#176d6b", value: (value) => `${fixed(value, 0)}t` }))}</div>`;
  }
  return "";
}
function liftOptimizationCharts() {
  if (state.page !== "optimization" || !state.plan) return;
  const chartNames = ["PPO episode rewards", "Cost versus resilience", "Strategy comparison"];
  const panels = chartNames.map((title) => [...$("#appView").querySelectorAll(".panel")]
    .find((item) => $(".panel-title", item)?.textContent.trim() === title)).filter(Boolean);
  const recommendations = [...$("#appView").querySelectorAll(".panel")]
    .find((item) => $(".panel-title", item)?.textContent.trim() === "Recommended interventions");
  const anchor = recommendations || $("#appView .page-heading");
  if (!anchor || panels.length < 2) return;
  const row = document.createElement("div");
  row.className = "layout-equal optimization-chart-row";
  anchor.insertAdjacentElement("afterend", row);
  for (const chart of panels) {
    const source = chart.parentElement;
    row.append(chart);
    if (source !== row && !source.querySelector(".panel")) source.remove();
  }
}
function riskTable(key = "vulnerability", label = "Climate vulnerability") {
  const rows = [...state.predictions].sort((a, b) => b[key] - a[key]);
  return `<div class="table-wrap"><table><thead><tr><th>Zone</th><th>${label}</th><th>Uncertainty</th><th>Band</th><th>Recommendation</th></tr></thead><tbody>${rows.map((prediction) => {
    const zone = getZone(prediction.zone_id), action = state.plan?.interventions?.find((item) => item.zone_id === prediction.zone_id);
    return `<tr data-zone="${esc(prediction.zone_id)}"><td><strong>${esc(zone?.name)}</strong><span class="zone-id">${esc(prediction.zone_id)}</span></td><td><div style="display:flex;align-items:center;gap:8px"><span class="mono">${pct(prediction[key])}</span><div class="bar-track"><div class="bar-fill" style="width:${Math.min(100, prediction[key] * 100)}%;--bar-color:${riskColor(prediction[key])}"></div></div></div></td><td class="mono">${pct(prediction.uncertainty)}</td><td><span class="risk-pill ${riskClass(prediction.risk_band)}">${esc(prediction.risk_band)}</span></td><td>${action ? esc(action.action) : "—"}</td></tr>`;
  }).join("")}</tbody></table></div>`;
}
function detailMarkup(zoneId) {
  const zone = getZone(zoneId), prediction = getPrediction(zoneId);
  if (!zone || !prediction) return "";
  const intervention = state.plan?.interventions?.find((item) => item.zone_id === zoneId);
  return `<div class="zone-detail"><h3>${esc(zone.name)} <span class="zone-id">${esc(zone.zone_id)} · ${esc(zone.features.land_use)}</span></h3><p>Flood ${pct(prediction.flood_risk)} · Heat ${pct(prediction.heat_risk)} · Vulnerability ${pct(prediction.vulnerability)}</p><p>Uncertainty ${pct(prediction.uncertainty)} · Confidence ${pct(prediction.confidence)} · Reliability ${pct(prediction.reliability)}</p><p>${intervention ? `${esc(intervention.action)} · install ${money(intervention.installation_cost)} · 10y maintenance ${money(intervention.maintenance_cost)} · lifecycle ${money(intervention.lifecycle_cost)}` : "No intervention in the current plan."}</p><p>${intervention ? `Flood -${pct(intervention.flood_reduction)} · heat -${pct(intervention.heat_reduction)} · carbon ${fixed(intervention.carbon_tonnes, 0)} t` : ""}</p></div>`;
}
function selectedTable(key) {
  const actions = state.plan?.interventions || [];
  if (!actions.length) return `<div class="empty-state">Run RL optimization to generate a zone-level intervention plan.</div>`;
  const sorted = key ? [...state.predictions].sort((a, b) => b[key] - a[key]) : state.predictions;
  const order = new Map(sorted.map((row, index) => [row.zone_id, index]));
  const rows = [...actions].sort((a, b) => order.get(a.zone_id) - order.get(b.zone_id));
  return `<div class="table-wrap"><table><thead><tr><th>Zone</th><th>Intervention</th><th>Install / maintenance (10y)</th><th>Flood reduction</th><th>Heat reduction</th><th>Carbon</th><th>Confidence</th></tr></thead><tbody>${rows.map((item) => `<tr data-zone="${esc(item.zone_id)}"><td>${esc(getZone(item.zone_id)?.name)}<span class="zone-id">${esc(item.zone_id)}</span></td><td>${esc(item.action)}</td><td>${money(item.installation_cost)} / ${money(item.maintenance_cost)}</td><td>${pct(item.flood_reduction)}</td><td>${pct(item.heat_reduction)}</td><td>${fixed(item.carbon_tonnes, 0)} t</td><td>${pct(item.prediction_confidence)}</td></tr>`).join("")}</tbody></table></div>`;
}
function overviewPage() {
  const plan = state.plan;
  const high = state.predictions.filter((item) => item.vulnerability >= .68).length;
  const cards = [
    ["Monitored zones", state.zones.length, "active nodes", "var(--green)"],
    ["Mean flood risk", pct(avg("flood_risk")), "model estimate", "var(--yellow)"],
    ["Mean heat risk", pct(avg("heat_risk")), "model estimate", "var(--violet)"],
    ["High-risk zones", high, "vulnerability ≥ 0.68", "var(--yellow)"],
    ["Mean uncertainty", pct(avg("uncertainty")), "MC dropout spread", "var(--violet)"],
    ["Interventions", plan?.intervention_count ?? "—", "latest plan", "var(--green)"],
    ["Lifecycle cost", plan ? money(plan.total_cost) : "—", "install + 10-year maintenance", "var(--yellow)"],
    ["Carbon benefit", plan ? `${fixed(plan.carbon_tonnes, 0)} t` : "—", "assumption-based", "var(--green)"],
    ["Flood risk reduction", plan ? `${fixed(plan.flood_reduction_pct, 1)}%` : "—", "relative · estimated", "var(--green)"],
    ["Heat risk reduction", plan ? `${fixed(plan.heat_reduction_pct, 1)}%` : "—", "relative · estimated", "var(--violet)"],
  ].map((item) => metricCard(...item)).join("");
  const action = plan ? `<button class="button button-outline" data-action="download">↓ Export report</button>` : `<button class="button button-primary" data-page="optimization">Run optimization</button>`;
  const ranked = [...state.predictions].sort((a, b) => b.vulnerability - a.vulnerability).slice(0, 5);
  return `${header("Climate resilience / <span style='color:var(--yellow)'>urban systems</span>", "CITY ADAPTATION INTELLIGENCE", "A spatial decision workspace for uncertainty-aware climate risk and green infrastructure.", action)}
    <div class="kpi-grid">${cards}<article class="kpi" style="--kpi-color:var(--violet)"><div class="kpi-label">Resilience score</div><div class="kpi-value">${plan ? `${fixed(plan.resilience_after, 1)}` : "—"}<small style="font-size:11px;color:var(--muted)"> / 100</small></div><div class="kpi-note">${plan ? `from ${fixed(plan.resilience_before, 1)} · estimated` : "run a plan to estimate"}</div></article></div>
    <div class="layout-two"><div>${mapPanel()}</div>${panel("Highest exposure", "RANKED BY OVERALL VULNERABILITY", `<div class="zone-list">${ranked.map((item) => { const zone = getZone(item.zone_id); return `<div class="zone-row" data-zone="${esc(item.zone_id)}"><div class="zone-name">${esc(zone.name)}<span class="zone-id">${esc(item.zone_id)} · ${pct(item.uncertainty)} uncertainty</span></div><span class="risk-pill ${riskClass(item.risk_band)}">${pct(item.vulnerability)}</span></div>`; }).join("")}</div>${state.plan ? `<div class="small-note" style="margin-top:10px">${esc(state.plan.policy_note)}</div>` : ""}`)}</div>
    <div class="layout-equal">${panel("Climate exposure by zone", "FLOOD RISK · CURRENT SCENARIO", sparkBars([...state.predictions].sort((a,b)=>b.flood_risk-a.flood_risk).slice(0,8), "flood_risk", { label: (row) => row.zone_id, color: (row) => riskColor(row.flood_risk) }))}${panel("Scenario envelope", "MEAN PREDICTED VULNERABILITY", sparkBars(state.scenarios, "vulnerability", { label: (row) => row.scenario, color: () => "#b6a0ff", value: (value) => pct(value) }))}</div>`;
}
function optimizationPage() {
  const plan = state.plan;
  const controls = `<form id="optimizationForm"><div class="control-grid">
    <div class="field"><label for="budgetInput">Total budget <span id="budgetValue">$500,000</span></label><input id="budgetInput" type="range" min="50000" max="1500000" step="25000" value="500000"></div>
    <div class="field"><label for="greenTarget">Green coverage target <span id="greenValue">35%</span></label><input id="greenTarget" type="range" min="10" max="75" step="1" value="35"></div>
    ${[["flood", "Flood risk priority", 40], ["heat", "Heat risk priority", 35], ["carbon", "Carbon priority", 15], ["green", "Green coverage priority", 10]].map(([key,label,value]) => `<div class="field"><label for="weight-${key}">${label} <span id="weightValue-${key}">${value}%</span></label><input class="weight-input" id="weight-${key}" data-weight="${key}" type="range" min="0" max="100" value="${value}"></div>`).join("")}
    <div class="field"><label for="episodes">PPO training episodes <span>advanced</span></label><input id="episodes" type="number" min="1" max="300" value="90"></div>
    <div class="field"><label for="scenarioOpt">Climate scenario</label><select id="scenarioOpt">${state.scenarios.map((item) => `<option ${item.scenario === $("#globalScenario").value ? "selected" : ""}>${esc(item.scenario)}</option>`).join("")}</select></div>
  </div><div class="control-footer"><span class="small-note">Multi-objective weights are normalized before training. Installation effects and costs are planning assumptions.</span><button class="button button-primary" id="runOptimize" type="submit">Run PPO planning</button></div></form>`;
  const results = plan ? `<div class="result-strip">${[["Selected actions", plan.intervention_count], ["Spend", money(plan.total_cost)], ["Resilience", `${fixed(plan.resilience_after,1)} / 100`], ["Carbon", `${fixed(plan.carbon_tonnes,0)} t`]].map(([label,value])=>`<div class="result-stat"><span>${label}</span><strong>${value}</strong></div>`).join("")}</div><div class="small-note">${esc(plan.policy_note)} Budget remaining: ${money(plan.budget_remaining)}. PPO reward is a training diagnostic, not a real-world outcome measure.</div>${panel("Recommended interventions", "SELECTED STRATEGY · ZONE-LEVEL BENEFITS", selectedTable("vulnerability"))}<div class="layout-equal">${panel("Strategy comparison", "SAME BUDGET · SAME RISK AND BENEFIT ASSUMPTIONS", sparkBars(Object.entries(plan.comparisons).map(([name,value])=>({name, resilience_after:value.resilience_after})), "resilience_after", {label: row=>row.name, value: value=>fixed(value,1), color: row=>row.name.startsWith("PPO")?"#d9ff54":"#7fd0a2"}))}${panel("PPO episode rewards", "CUMULATIVE EPISODE REWARD · TRAINING TRACE", sparkBars(plan.training_rewards.slice(-18).map((reward,index)=>({name:`E${index+1}`,reward})), "reward", {label:row=>row.name,color:()=>"#b6a0ff"}))}</div>` : `<div class="empty-state">Configure budget and objective weights, then train a PPO planning policy.</div>`;
  return `${header("Intervention policy", "REINFORCEMENT LEARNING", "A constrained, multi-objective planning exercise across the current zone graph.")}<div class="layout-two"><div class="panel"><div class="panel-head"><div><div class="panel-title">Optimization controls</div><div class="panel-subtitle">BUDGET · PRIORITIES · CLIMATE STRESS</div></div></div>${controls}</div><div>${mapPanel()}<div class="callout">Policy output is compared to seeded-random and risk-weighted greedy baselines. Planning effects are illustrative assumptions, not calibrated impact estimates.</div></div></div><div style="margin-top:10px">${results}</div>`;
}
function scenariosPage() {
  const chart = sparkBars(state.scenarios, "vulnerability", { label: row=>row.scenario, color:()=>"#b6a0ff", value:value=>pct(value) });
  const cards = state.scenarios.map((row) => `<article class="scenario-item"><strong>${esc(row.scenario)}</strong><span>MEAN FLOOD ESTIMATE</span><b>${pct(row.flood_risk)}</b><span>MEAN HEAT / UNCERTAINTY</span><div style="margin-top:5px;font-size:11px">${pct(row.heat_risk)} <span style="color:#819080">/</span> ${pct(row.uncertainty)}</div></article>`).join("");
  return `${header("Stress-test the study area", "SCENARIO ANALYSIS", "Same zones, perturbed climate conditions. Scenario differences are simulated, not forecasts.")}<div class="scenario-grid">${cards}</div><div class="layout-two">${panel("Vulnerability by scenario", "MEAN MODEL ESTIMATE", chart)}${panel("Scenario definitions", "INPUT PERTURBATIONS", `<div class="zone-list">${[["Baseline","Observed sample values as supplied in the synthetic dataset."],["Heavy rainfall","Rainfall inputs scaled by 1.25."],["Extreme rainfall","Rainfall inputs scaled by 1.55."],["Heatwave","Temperature inputs increased by 3.5 °C."],["Compound stress","Rainfall scaled by 1.35 with a +2.4 °C temperature perturbation."]].map(([name,desc])=>`<div class="zone-row"><div class="zone-name">${name}<span class="zone-id">${desc}</span></div></div>`).join("")}</div>`)}</div>`;
}
function resultGraphsPage() {
  return `${header(...pages["result-graphs"], `<button class="button button-outline" data-action="generate-figures">↻ Regenerate figures</button>`)}<div class="figure-provenance" id="figureProvenance">Figures are generated from current predictions and saved experiments. They are not illustrative stock plots.</div><div class="figure-grid" id="figureGrid"><div class="loading-state"><span class="spinner"></span>Generating 1200-DPI figures from the active dataset…</div></div>`;
}
function pageContent() {
  const predictionKey = state.page === "flood" ? "flood_risk" : state.page === "heat" ? "heat_risk" : state.page === "uncertainty" ? "uncertainty" : "vulnerability";
  const title = pages[state.page];
  if (state.page === "overview") return overviewPage();
  if (state.page === "optimization") return optimizationPage();
  if (state.page === "scenarios") return scenariosPage();
  if (state.page === "result-graphs") return resultGraphsPage();
  if (state.page === "map") return `${header(...title)}${mapPanel()}<div style="margin-top:10px">${panel("Zone inventory", "RISK, MODEL SPREAD AND CURRENT PLAN", riskTable())}</div>`;
  if (["flood", "heat", "uncertainty"].includes(state.page)) {
    const label = state.page === "flood" ? "Flood probability" : state.page === "heat" ? "Heat probability" : "Uncertainty score";
    const sorted = [...state.predictions].sort((a,b)=>b[predictionKey]-a[predictionKey]);
    return `${header(...title)}<div class="layout-two">${mapPanel()}${panel(`Highest ${label.toLowerCase()}`, "SELECT A ZONE TO INSPECT", `<div class="zone-list">${sorted.slice(0,8).map((row)=>`<div class="zone-row" data-zone="${esc(row.zone_id)}"><div class="zone-name">${esc(getZone(row.zone_id)?.name)}<span class="zone-id">${esc(row.zone_id)} · ${pct(row.uncertainty)} uncertainty</span></div><span class="risk-pill ${riskClass(row.risk_band)}">${pct(row[predictionKey])}</span></div>`).join("")}</div>`)}</div><div style="margin-top:10px">${panel("Zone-level estimates", "MONTE CARLO DROPOUT · 32 DRAWS", riskTable(predictionKey,label))}</div>`;
  }
  if (state.page === "recommendations") return `${header(...title, state.plan ? `<button class="button button-outline" data-action="download">↓ Export report</button>` : "")}${mapPanel()}<div style="margin-top:10px">${panel("Intervention schedule", "ESTIMATED COST · RISK CHANGE · CARBON CO-BENEFIT", selectedTable())}</div>`;
  if (state.page === "comparison") {
    const plan = state.plan;
    if (!plan) return `${header(...title)}<div class="panel empty-state">Run an optimization to calculate before-and-after estimates.</div>`;
    const metrics = [["Flood risk",plan.flood_before,plan.flood_after],["Heat risk",plan.heat_before,plan.heat_after],["Green coverage",plan.green_coverage_before/100,plan.green_coverage_after/100],["Resilience score",plan.resilience_before/100,plan.resilience_after/100]];
    return `${header(...title)}<div class="layout-equal">${metrics.map(([label,before,after])=>panel(label,"BASELINE → PLANNED ESTIMATE",`<div class="result-strip" style="grid-template-columns:1fr 1fr"><div class="result-stat"><span>Before</span><strong>${label.includes("score")?fixed(before*100,1):pct(before,1)}</strong></div><div class="result-stat"><span>After</span><strong style="color:var(--yellow)">${label.includes("score")?fixed(after*100,1):pct(after,1)}</strong></div></div><div class="bar-track" style="height:10px"><div class="bar-fill" style="width:${Math.min(after*100,100)}%;--bar-color:#b6a0ff"></div></div>`)).join("")}</div><div class="layout-two">${panel("Strategy resilience", "BASELINE PLAN COMPARISON", sparkBars(Object.entries(plan.comparisons).map(([name,value])=>({name,resilience_after:value.resilience_after})),"resilience_after",{label:row=>row.name,value:v=>fixed(v,1),color:row=>row.name.startsWith("PPO")?"#d9ff54":"#7fd0a2"}))}${panel("Expected co-benefits", "AGGREGATED PLANNING ASSUMPTIONS", `<div class="result-strip">${[["Flood reduction",`${fixed(plan.flood_reduction_pct,1)}%`],["Heat reduction",`${fixed(plan.heat_reduction_pct,1)}%`],["Carbon",`${fixed(plan.carbon_tonnes,0)} t`],["Green gain",`${fixed(plan.green_gain_pct_points,1)} pp`]].map(([label,value])=>`<div class="result-stat"><span>${label}</span><strong>${value}</strong></div>`).join("")}</div><p class="small-note">Estimated outcomes are based on fixed intervention-effect assumptions applied to model risk scores. They are not causal or field-validated impacts.</p>`)}</div>`;
  }
  if (state.page === "explain") return `${header(...title)}<div class="layout-two"><section class="panel"><div class="panel-head"><div><div class="panel-title">Zone attribution</div><div class="panel-subtitle">INTEGRATED GRADIENTS · MODEL INPUT CONTRIBUTIONS</div></div></div><div class="panel-body"><div class="field"><label for="explainZone">Urban zone</label><select id="explainZone">${state.zones.map((zone)=>`<option value="${esc(zone.zone_id)}" ${zone.zone_id===state.selectedZone?"selected":""}>${esc(zone.name)} · ${esc(zone.zone_id)}</option>`).join("")}</select></div><div id="explainOutput" style="margin-top:15px"><div class="empty-state">Select a zone to calculate attribution values.</div></div></div></section>${mapPanel()}</div>`;
  if (state.page === "analytics") {
    const data = state.analytics;
    const ablations = data?.ablations || [];
    return `${header(...title)}<div class="result-strip">${[["Fit MAE",fixed(data?.model?.mae,3)],["Fit RMSE",fixed(data?.model?.rmse,3)],["Proxy R²",fixed(data?.model?.r2_proxy_fit,3)],["Saved experiments",data?.history?.length||0]].map(([label,value])=>`<div class="result-stat"><span>${label}</span><strong>${value}</strong></div>`).join("")}</div><div class="callout">${esc(data?.model?.label_note||"Metrics are fit diagnostics against generated proxy labels; they are not held-out or field-validation results.")}</div><div class="layout-two">${panel("Architecture ablations", "IN-SAMPLE ERROR AGAINST SYNTHETIC PROXY LABELS", `<div class="table-wrap"><table><thead><tr><th>Variant</th><th>MAE</th><th>RMSE</th><th>Interpretation</th></tr></thead><tbody>${ablations.map(row=>`<tr><td>${esc(row.name)}</td><td>${fixed(row.mae,4)}</td><td>${fixed(row.rmse,4)}</td><td>${row.uncertainty===0?"Uncertainty output disabled":row.cost?`Cost ${money(row.cost)}`:"Risk fit diagnostic"}</td></tr>`).join("")}</tbody></table></div>`)}${panel("Risk by zone", "VULNERABILITY PROBABILITY", sparkBars([...state.predictions].sort((a,b)=>b.vulnerability-a.vulnerability).slice(0,8),"vulnerability",{label:row=>row.zone_id,color:row=>riskColor(row.vulnerability)}))}</div><div style="margin-top:10px">${panel("Saved experiments", "LOCAL SQLITE HISTORY", data?.history?.length ? `<div class="table-wrap"><table><thead><tr><th>Run</th><th>Type</th><th>Created</th><th>Cost</th><th>Interventions</th></tr></thead><tbody>${data.history.map(row=>{let metrics={};try{metrics=JSON.parse(row.metrics_json)}catch{}return `<tr><td class="mono">${esc(row.run_id.slice(0,8))}</td><td>${esc(row.kind)}</td><td>${esc(row.created_at)}</td><td>${money(metrics.total_cost)}</td><td>${metrics.intervention_count||0}</td></tr>`}).join("")}</tbody></table></div>`:`<div class="empty-state">Run an optimization to save an experiment.</div>`)}</div>`;
  }
  if (state.page === "system") return `${header(...title)}<div class="panel"><div class="panel-head"><div><div class="panel-title">Research workflow</div><div class="panel-subtitle">END-TO-END PROTOTYPE PIPELINE</div></div><span class="risk-pill elevated">SIMULATION</span></div><div class="panel-body"><div class="architecture">${["Data ingestion","Spatial graph","Spatio-temporal GNN","Risk prediction","MC dropout uncertainty","PPO planning","Infrastructure plan","Integrated gradients","Resilience dashboard"].map((name,index)=>`${index?`<span class="architecture-arrow">→</span>`:""}<div class="architecture-node"><b>${String(index+1).padStart(2,"0")}</b><span>${name}</span></div>`).join("")}</div><div class="callout">The included 12-zone dataset and climate observations are generated demonstration inputs. No real city observations, measured treatment effects or external validation are bundled. Replace the data and calibrate all intervention assumptions before operational use.</div></div></div><div class="layout-two"><div>${panel("Training controls", "ADVANCED · SYNTHETIC LABEL FIT", `<form id="trainingForm"><div class="control-grid"><div class="field"><label for="trainEpochs">Epochs</label><input id="trainEpochs" type="number" min="10" max="500" value="110"></div><div class="field"><label for="hiddenDim">Hidden dimension</label><input id="hiddenDim" type="number" min="8" max="128" value="24"></div><div class="field"><label for="learningRate">Learning rate</label><input id="learningRate" type="number" min="0.0001" max="0.1" step="0.001" value="0.012"></div></div><div class="control-footer"><span class="small-note">Deterministic seed · PyTorch CPU</span><button id="trainButton" class="button button-outline" type="submit">Train risk model</button></div></form>`)}${panel("Dataset", "CURRENT STUDY AREA", `<p class="small-note">${state.zones.length} zones · 6 seeded climate time steps per zone · CSV schema available in <span class="mono">data/sample_zones.csv</span>.</p><p class="small-note">Accepted upload: CSV with zone ID/name, coordinates, climate, elevation, green cover, density, drainage, imperviousness, flood history and land-use columns.</p><button class="button button-outline" data-action="upload">↑ Choose CSV</button>`)}</div>${panel("Local REST API", "OPENAPI-READY ROUTES", `<div class="zone-list">${[["GET","/api/health"],["GET","/api/overview"],["GET","/api/graph"],["POST","/api/predict"],["POST","/api/optimize"],["GET","/api/scenarios"],["GET","/api/explain/{zone_id}"],["GET","/api/analytics"],["POST","/api/upload"],["POST","/api/train"],["GET","/api/report/{run_id}"]].map(([verb,path])=>`<div class="zone-row"><span class="mono" style="font-size:10px;color:var(--green)">${verb}</span><span class="mono" style="font-size:10px">${path}</span></div>`).join("")}</div>`)}</div>`;
  return `${header(...title)}`;
}

async function showExplain(zoneId) {
  const target = $("#explainOutput"); if (!target) return;
  target.innerHTML = `<div class="loading-state"><span class="spinner"></span>Calculating integrated gradients…</div>`;
  try {
    const data = await api(`/api/explain/${encodeURIComponent(zoneId)}`);
    target.innerHTML = `<div class="result-strip" style="grid-template-columns:repeat(3,minmax(0,1fr))"><div class="result-stat"><span>Flood risk</span><strong>${pct(data.prediction.flood_risk)}</strong></div><div class="result-stat"><span>Heat risk</span><strong>${pct(data.prediction.heat_risk)}</strong></div><div class="result-stat"><span>Confidence</span><strong>${pct(data.prediction.confidence)}</strong></div></div><div style="margin:14px 0">${data.contributions.map((item)=>`<div style="display:grid;grid-template-columns:minmax(100px,1fr) 2fr 48px;gap:8px;align-items:center;margin:11px 0;font-size:10px"><span>${esc(item.feature)}</span><div class="bar-track"><div class="bar-fill" style="width:${Math.min(100,Math.abs(item.contribution)*6000)}%;--bar-color:${item.contribution<0?"#b6a0ff":"#d9ff54"}"></div></div><span class="mono">${item.contribution>0?"+":""}${fixed(item.contribution,3)}</span></div>`).join("")}</div><div class="small-note">${esc(data.method)} Attribution values are signed local sensitivities in normalized input space, not causal effects.</div>${data.recommendation?`<div class="zone-detail"><h3>${esc(data.recommendation.action)}</h3><p>Estimated cost ${money(data.recommendation.cost)} · flood change ${pct(data.recommendation.flood_reduction)} · heat change ${pct(data.recommendation.heat_reduction)}</p><p>Carbon assumption ${fixed(data.recommendation.carbon_tonnes,0)} t · prediction confidence ${pct(data.recommendation.prediction_confidence)}</p></div>`:`<div class="empty-state">No current plan recommendation for this zone.</div>`}`;
    target.innerHTML = deepPaletteMarkup(target.innerHTML);
  } catch (error) { target.innerHTML = `<div class="callout">${esc(error.message)}</div>`; }
}
async function loadFigures() {
  const grid = $("#figureGrid");
  if (!grid) return;
  grid.innerHTML = `<div class="loading-state"><span class="spinner"></span>Rendering publication figures at 1200 DPI…</div>`;
  try {
    const result = await api(`/api/figures?scenario=${encodeURIComponent($("#globalScenario").value)}`);
    $("#figureProvenance").textContent = `${result.figures.length} figures · ${result.dpi} DPI · ${result.model} · ${result.data_status.name} · generated ${new Date(result.generated_at).toLocaleTimeString()}. Proxy-label and intervention-assumption limitations apply.`;
    grid.innerHTML = result.figures.map((figure, index) => `<article class="figure-card"><div class="figure-card-head"><div><span class="figure-index">FIG ${String(index + 1).padStart(2, "0")}</span><h2>${esc(figure.title)}</h2></div><a class="button button-outline figure-download" href="${esc(figure.url)}?download=1" download="${esc(figure.filename)}" title="Download ${esc(figure.title)} at 350 DPI">↓ PNG</a></div><a class="figure-image-link" href="${esc(figure.url)}?v=${encodeURIComponent(result.generated_at)}" target="_blank" rel="noreferrer"><img src="${esc(figure.url)}?v=${encodeURIComponent(result.generated_at)}" alt="${esc(figure.title)}. ${esc(figure.caption)}" loading="lazy"></a><p class="figure-caption">${esc(figure.caption)}</p><div class="figure-meta">${figure.width_px} × ${figure.height_px} px · ${figure.dpi} DPI · PNG</div></article>`).join("");
    grid.querySelectorAll(".figure-card").forEach((card, index) => {
      const figure = result.figures[index];
      const pngLink = card.querySelector(".figure-download");
      pngLink.title = `Download ${figure.title} at ${figure.dpi} DPI`;
      const pdfLink = document.createElement("a");
      pdfLink.className = "button button-outline figure-download";
      pdfLink.href = `${figure.pdf_url}?download=1`;
      pdfLink.download = figure.pdf_filename;
      pdfLink.title = `Download ${figure.title} as a vector PDF`;
      pdfLink.textContent = "PDF";
      card.querySelector(".figure-card-head").append(pdfLink);
      card.querySelector(".figure-meta").textContent = `${figure.width_px} × ${figure.height_px} px · ${figure.dpi} DPI · PNG + vector PDF`;
    });
  } catch (error) {
    grid.innerHTML = `<div class="callout">Figure generation failed: ${esc(error.message)}</div>`;
  }
}

async function render() {
  const current = pages[state.page]; $("#pageName").textContent = current[0];
  document.title = `${current[0]} · ${researchTitle}`;
  document.querySelectorAll(".nav-item").forEach((button)=>button.classList.toggle("active",button.dataset.page===state.page));
  $("#appView").innerHTML = deepPaletteMarkup(pageContent());
  if (state.page === "system") mountModelSelector();
  if (state.selectedZone && $("#zoneDetail")) $("#zoneDetail").innerHTML = detailMarkup(state.selectedZone);
  if (state.page === "explain") showExplain(state.selectedZone || state.zones[0]?.zone_id);
  if (state.page === "analytics") {
    try { state.analytics = await api("/api/analytics"); $("#appView").innerHTML = deepPaletteMarkup(pageContent() + additionalCharts()); }
    catch (error) { toast(error.message, true); }
  } else if (state.page === "result-graphs") loadFigures();
  else if (state.page !== "explain") {
    $("#appView").insertAdjacentHTML("beforeend", deepPaletteMarkup(additionalCharts()));
    liftOptimizationCharts();
  }
  else if (state.page !== "explain") $("#appView").insertAdjacentHTML("beforeend", additionalCharts());
}
async function reloadScenario(scenario) {
  try {
    const data = await api(`/api/overview?scenario=${encodeURIComponent(scenario)}`);
    state.zones = data.zones; state.predictions = data.predictions; state.dataset = data.dataset;
    state.plan = data.latest_plan?.scenario === scenario ? data.latest_plan : null;
    state.graph = await api("/api/graph"); updateFreshness(data.computed_at); updateDatasetBadge(data.dataset); render();
  } catch (error) { toast(error.message, true); }
}
async function refreshLive(announce = false) {
  try {
    const scenario = $("#globalScenario").value;
    const data = await api(`/api/live?scenario=${encodeURIComponent(scenario)}`);
    state.predictions = data.predictions;
    state.dataset = data.dataset;
    updateFreshness(data.computed_at);
    updateDatasetBadge(data.dataset);
    if (state.page === "result-graphs") await loadFigures();
    else if (!["optimization", "system"].includes(state.page)) await render();
    if (announce) toast(`Values refreshed from ${data.dataset.name}; source remains ${data.dataset.source_type}.`);
  } catch (error) {
    if (announce) toast(error.message, true);
  }
}
async function loadInitial() {
  try {
    const [overview, scenarios, graph, health] = await Promise.all([api("/api/overview"), api("/api/scenarios"), api("/api/graph"), api("/api/health")]);
    state.zones = overview.zones; state.predictions = overview.predictions; state.plan = overview.latest_plan;
    state.dataset = overview.dataset; state.scenarios = scenarios; state.graph = graph;
    state.models = health.models; state.activeModel = health.model_type; state.device = health.device;
    updateDatasetBadge(state.dataset); updateFreshness(overview.computed_at);
    await render();
  } catch (error) { $("#appView").innerHTML = `<div class="callout">Unable to initialize workspace: ${esc(error.message)}</div>`; }
}

document.body.addEventListener("click", async (event) => {
  const page = event.target.closest("[data-page]");
  if (page) { event.preventDefault(); state.page = page.dataset.page; await render(); return; }
  const mode = event.target.closest("[data-map-mode]");
  if (mode) { state.mapMode = mode.dataset.mapMode; await render(); return; }
  const zone = event.target.closest("[data-zone]");
  if (zone) { state.selectedZone = zone.dataset.zone; if ($("#zoneDetail")) $("#zoneDetail").innerHTML = detailMarkup(state.selectedZone); if (state.page === "explain") showExplain(state.selectedZone); return; }
  if (event.target.closest("[data-action='upload']") || event.target.closest("#uploadButton")) $("#fileInput").click();
  if (event.target.closest("#refreshButton")) refreshLive(true);
  if (event.target.closest("[data-action='generate-figures']")) loadFigures();
  if (event.target.closest("[data-action='download']")) {
    if (!state.plan?.run_id) return toast("Run an optimization before exporting a plan.", true);
    window.location.href = `/api/report/${encodeURIComponent(state.plan.run_id)}`;
  }
});
$("#globalScenario").addEventListener("change", (event) => reloadScenario(event.target.value));
$("#fileInput").addEventListener("change", async (event) => {
  const file = event.target.files[0]; if (!file) return;
  const form = new FormData(); form.append("file", file);
  toast("Validating columns, loading zones and retraining the risk model…");
  try {
    const result = await api("/api/upload", { method: "POST", body: form });
    await loadInitial(); toast(`${result.zones} zones loaded from ${result.dataset}. ${result.note}`);
  } catch (error) { toast(error.message, true); }
  event.target.value = "";
});
document.body.addEventListener("input", (event) => {
  if (event.target.id === "budgetInput") $("#budgetValue").textContent = money(event.target.value);
  if (event.target.id === "greenTarget") $("#greenValue").textContent = `${event.target.value}%`;
  if (event.target.matches(".weight-input")) $(`#weightValue-${event.target.dataset.weight}`).textContent = `${event.target.value}%`;
});
document.body.addEventListener("change", (event) => {
  if (event.target.id === "explainZone") { state.selectedZone = event.target.value; showExplain(event.target.value); }
});
document.body.addEventListener("submit", async (event) => {
  if (event.target.id === "optimizationForm") {
    event.preventDefault(); const button = $("#runOptimize"); button.disabled = true; button.innerHTML = `<span class="spinner" style="display:inline-block;vertical-align:middle"></span> Training PPO…`;
    try {
      const body = { budget: Number($("#budgetInput").value), green_target: Number($("#greenTarget").value),
        flood_weight: Number($("#weight-flood").value), heat_weight: Number($("#weight-heat").value),
        carbon_weight: Number($("#weight-carbon").value), green_weight: Number($("#weight-green").value),
        episodes: Number($("#episodes").value), scenario: $("#scenarioOpt").value };
      const result = await api("/api/optimize", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      state.plan = result; state.zones = result.zones; state.predictions = result.predictions; state.page = "optimization"; await render(); toast(result.policy_note);
    } catch (error) { toast(error.message, true); button.disabled = false; button.textContent = "Run PPO planning"; }
  }
  if (event.target.id === "trainingForm") {
    event.preventDefault(); const button = $("#trainButton"); button.disabled = true; button.textContent = "Training model…";
    try {
      const result = await api("/api/train", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ epochs: Number($("#trainEpochs").value), hidden_dim: Number($("#hiddenDim").value), learning_rate: Number($("#learningRate").value), model_type: $("#modelType")?.value || state.activeModel }) });
      state.activeModel = result.model_type;
      toast(`${result.metrics.model_name} trained. In-sample proxy MAE ${fixed(result.metrics.mae,4)}; not field validation.`); await reloadScenario($("#globalScenario").value);
    } catch (error) { toast(error.message, true); button.disabled = false; button.textContent = "Train risk model"; }
  }
});
loadInitial();
window.setInterval(() => refreshLive(false), 30000);