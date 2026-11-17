/* ragmap front-end: d3 force graph + passage panel + question answering. */
const palette = ["#6aa9ff", "#ff8fa3", "#7bd389", "#ffb86b", "#c39bff", "#5fd4d4", "#f2e86d", "#ff7a5c", "#a0b4ff", "#9ad18b"];
const state = { docId: null, map: null, chunks: new Map(), hidden: new Set(), selected: null, hits: new Set() };
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const color = (c) => palette[c % palette.length];

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

async function refreshDocs(selectId) {
  const docs = await api("/api/documents");
  const sel = $("#docSelect");
  sel.innerHTML = docs.map((d) => `<option value="${d.doc_id}">${esc(d.title)}</option>`).join("");
  sel.classList.toggle("hidden", !docs.length);
  const target = selectId || (docs[0] && docs[0].doc_id);
  if (target) { sel.value = target; await loadDoc(target); }
}

async function loadDoc(docId) {
  state.docId = docId; state.hidden.clear(); state.selected = null; state.hits.clear();
  state.map = await api(`/api/documents/${docId}/map`);
  const chunks = await api(`/api/documents/${docId}/chunks`);
  state.chunks = new Map(chunks.map((c) => [c.chunk_id, c]));
  $("#empty").classList.add("hidden");
  $("#docTitle").textContent = state.map.title;
  const s = state.map.stats;
  $("#stats").textContent = `${s.pages ? s.pages + " pages · " : ""}${s.words.toLocaleString()} words · ${s.chunks} passages · ${s.concepts} concepts · ${s.relations} relations · ${state.map.topics.length} topics`;
  $("#answer").innerHTML = "";
  renderLegend(); renderGraph(); showOverview();
}

function renderLegend() {
  $("#legend").innerHTML = state.map.topics.map((t) =>
    `<span class="chip" data-topic="${t.id}" title="${esc(t.concepts.join(", "))}"><span class="dot" style="background:${color(t.id)}"></span>${esc(t.label)}</span>`).join("");
  document.querySelectorAll("#legend .chip").forEach((el) => el.addEventListener("click", () => {
    const id = +el.dataset.topic;
    state.hidden.has(id) ? state.hidden.delete(id) : state.hidden.add(id);
    el.classList.toggle("off"); applyStyles();
  }));
}

let sim, nodeSel, linkSel;
function renderGraph() {
  const svg = d3.select("#graph"); svg.selectAll("*").remove();
  const { width, height } = svg.node().getBoundingClientRect();
  const nodes = state.map.nodes.map((d) => ({ ...d }));
  const links = state.map.links.map((d) => ({ ...d }));
  const maxScore = d3.max(nodes, (d) => d.score) || 1;
  const r = d3.scaleSqrt().domain([0, maxScore]).range([4, 22]);
  const g = svg.append("g");
  svg.call(d3.zoom().scaleExtent([0.3, 4]).on("zoom", (e) => g.attr("transform", e.transform)));

  linkSel = g.append("g").selectAll("line").data(links).join("line").attr("class", "link")
    .attr("stroke-width", (d) => 0.6 + 2.2 * d.weight);
  nodeSel = g.append("g").selectAll("g").data(nodes).join("g").attr("class", "node")
    .call(d3.drag().on("start", (e, d) => { if (!e.active) sim.alphaTarget(0.2).restart(); d.fx = d.x; d.fy = d.y; })
      .on("drag", (e, d) => { d.fx = e.x; d.fy = e.y; })
      .on("end", (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }))
    .on("click", (e, d) => selectConcept(d.id))
    .on("mouseenter", (e, d) => hover(d.id)).on("mouseleave", () => hover(null));
  nodeSel.append("circle").attr("r", (d) => r(d.score)).attr("fill", (d) => color(d.community));
  const labelCut = d3.quantile(nodes.map((d) => d.score).sort(d3.ascending), 0.45);
  nodeSel.append("text").text((d) => d.label).attr("class", (d) => (d.score >= labelCut ? "" : "minor")).attr("x", (d) => r(d.score) + 3).attr("y", 4)
    .style("font-size", (d) => `${10 + 5 * Math.sqrt(d.score / maxScore)}px`);
  nodeSel.append("title").text((d) => `${d.label} · ${d.count} mentions · ${d.chunks.length} passages`);

  sim = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id((d) => d.id).distance((d) => 105 - 40 * d.weight).strength(0.25))
    .force("charge", d3.forceManyBody().strength(-380))
    .force("collide", d3.forceCollide().radius((d) => r(d.score) + 14))
    .force("x", d3.forceX(width / 2).strength(0.045)).force("y", d3.forceY(height / 2 + 10).strength(0.07))
    .on("tick", () => {
      linkSel.attr("x1", (d) => d.source.x).attr("y1", (d) => d.source.y).attr("x2", (d) => d.target.x).attr("y2", (d) => d.target.y);
      nodeSel.attr("transform", (d) => `translate(${d.x},${d.y})`);
    });
  applyStyles();
}

function neighbors(id) {
  const s = new Set([id]);
  state.map.links.forEach((l) => { if (l.source === id) s.add(l.target); if (l.target === id) s.add(l.source); });
  return s;
}

function applyStyles(focus = null) {
  if (!nodeSel) return;
  const nb = focus !== null ? neighbors(focus) : null;
  const visible = (d) => !state.hidden.has(d.community);
  const hitMode = state.hits.size > 0;
  nodeSel.classed("dim", (d) => !visible(d) || (nb ? !nb.has(d.id) : hitMode && !state.hits.has(d.id)))
    .classed("hit", (d) => state.hits.has(d.id)).classed("selected", (d) => d.id === state.selected);
  const id = (x) => (typeof x === "object" ? x.id : x);
  linkSel.classed("dim", (l) => {
    const s = state.map.nodes[id(l.source)], t = state.map.nodes[id(l.target)];
    if (!visible(s) || !visible(t)) return true;
    if (nb) return !(nb.has(s.id) && nb.has(t.id) && (s.id === focus || t.id === focus));
    return hitMode && !(state.hits.has(s.id) && state.hits.has(t.id));
  }).classed("hot", (l) => hitMode && state.hits.has(id(l.source)) && state.hits.has(id(l.target)));
}
const hover = (id) => applyStyles(id ?? state.selected);

function highlight(text, label) {
  const words = label.split(/\s+/).map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/s$/i, ""));
  const re = new RegExp(`(${words.join("\\w*[\\s-]+")}\\w*)`, "gi");
  return esc(text).replace(re, "<mark>$1</mark>");
}

function passageHTML(c, label, extra = "") {
  const where = [c.section, c.page ? `p. ${c.page}` : ""].filter(Boolean).join(" · ");
  return `<div class="passage" data-chunk="${c.chunk_id}"><div class="where"><span>${esc(where || "passage")}</span><span>${extra}</span></div>${label ? highlight(c.text, label) : esc(c.text)}</div>`;
}

function selectConcept(id) {
  state.selected = id; applyStyles(id);
  const n = state.map.nodes[id];
  const rel = [...neighbors(id)].filter((x) => x !== id).map((x) => state.map.nodes[x]).sort((a, b) => b.score - a.score);
  const passages = n.chunks.map((cid) => state.chunks.get(cid)).filter(Boolean);
  $("#panel").innerHTML = `
    <h3><span class="dot" style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${color(n.community)}"></span> ${esc(n.label)}</h3>
    <div class="meta">${n.count} mentions · ${passages.length} passages${n.pages.length ? " · pages " + n.pages.join(", ") : ""}</div>
    <h3>Related concepts</h3><div class="related">${rel.map((r) => `<span class="chip" data-node="${r.id}"><span class="dot" style="background:${color(r.community)}"></span>${esc(r.label)}</span>`).join("") || '<span class="hint">none</span>'}</div>
    <h3>Source passages</h3>${passages.map((c) => passageHTML(c, n.label)).join("")}`;
  document.querySelectorAll("#panel .chip[data-node]").forEach((el) => el.addEventListener("click", () => selectConcept(+el.dataset.node)));
}

function showOverview() {
  const m = state.map;
  $("#panel").innerHTML = `<h3>Topics</h3>${m.topics.map((t) => `
      <div class="passage" data-topic="${t.id}"><div class="where"><span><span class="dot" style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${color(t.id)}"></span> ${esc(t.label)}</span><span>${t.size} concepts</span></div>
      ${esc(t.concepts.join(" · "))}${t.sections.length ? `<div class="meta">mostly in: ${esc(t.sections.join(", "))}</div>` : ""}</div>`).join("")}
    <h3>Outline</h3><div class="meta">${m.outline.map((o) => esc(o.section)).join(" → ")}</div>`;
  document.querySelectorAll("#panel .passage[data-topic]").forEach((el) => el.addEventListener("click", () => {
    const t = +el.dataset.topic; const top = m.nodes.filter((n) => n.community === t).sort((a, b) => b.score - a.score)[0];
    if (top) selectConcept(top.id);
  }));
}

$("#askForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = $("#query").value.trim(); if (!query || !state.docId) return;
  $("#answer").innerHTML = '<div class="meta">Searching…</div>';
  try {
    const res = await api("/api/search", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, k: 5, mode: $("#mode").value, doc_ids: [state.docId], answer: true }) });
    state.hits = new Set(res.concepts[state.docId] || []); state.selected = null; applyStyles();
    const a = res.answer;
    const t = res.timings;
    $("#answer").innerHTML = `<div class="answer-box">${esc(a.answer).replace(/\[(\d+)\]/g, "<b>[$1]</b>")}</div>
      <div class="meta">${esc(a.generator)} · retrieval ${t.total_ms.toFixed(0)} ms${t.rerank_ms ? ` (rerank ${t.rerank_ms.toFixed(0)} ms)` : ""} · ${state.hits.size} concepts lit on the map${a.warning ? " · " + esc(a.warning) : ""}</div>`;
    $("#panel").innerHTML = `<h3>Retrieved passages</h3>` + res.hits.map((h, i) =>
      passageHTML(h, null, `[${i + 1}] ${h.scores.rerank !== undefined ? "ce " + h.scores.rerank.toFixed(2) : "score " + h.score.toFixed(3)}`)).join("");
  } catch (err) { $("#answer").innerHTML = `<div class="meta">Error: ${esc(err.message)}</div>`; }
});

$("#fileInput").addEventListener("change", async (e) => {
  const file = e.target.files[0]; if (!file) return;
  const fd = new FormData(); fd.append("file", file);
  $("#busy").classList.remove("hidden");
  try { const m = await api("/api/documents", { method: "POST", body: fd }); await refreshDocs(m.doc_id); }
  catch (err) { alert(`Upload failed: ${err.message}`); }
  finally { $("#busy").classList.add("hidden"); e.target.value = ""; }
});
$("#docSelect").addEventListener("change", (e) => loadDoc(e.target.value));
window.addEventListener("resize", () => state.map && renderGraph());
refreshDocs().catch(() => {});
