const $ = id => document.getElementById(id);
let allDocs = [];
let chosenFile = null;

const TYPE_LABEL = { page: "Page", pdf: "PDF", upload: "Upload", image: "Image", video: "Video", audio: "Audio", document: "Doc" };

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined && text !== null) e.textContent = text;
  return e;
}

function link(text, href, cls) {
  if (!href) return el("span", cls, text);
  const a = el("a", cls, text);
  a.href = href; a.target = "_blank"; a.rel = "noopener";
  return a;
}

function renderDocs() {
  const f = $("filter").value.toLowerCase();
  const body = $("docs");
  body.textContent = "";
  const shown = allDocs.filter(d => (d.title + " " + d.source_type).toLowerCase().includes(f));
  $("docCount").textContent = f ? `${shown.length} of ${allDocs.length}` : `${allDocs.length} total`;
  if (!shown.length) {
    const tr = el("tr"); const td = el("td", "muted"); td.colSpan = 5;
    td.textContent = allDocs.length ? "No match." : "Nothing indexed yet. Run python run_pipeline.py or upload a file.";
    tr.appendChild(td); body.appendChild(tr);
    return;
  }
  shown.forEach(d => {
    const tr = el("tr");
    const name = el("td", "name");
    name.appendChild(link(d.title, d.source_url.startsWith("http") ? d.source_url : d.saved_copy));
    name.title = d.title;
    const type = el("td"); type.appendChild(el("span", "tag t-" + d.source_type, TYPE_LABEL[d.source_type] || d.source_type));
    const date = el("td", "muted", d.date && d.date !== "undated" ? d.date : "–");
    const n = el("td", "r", d.chunks);
    const act = el("td", "r");
    if (d.deletable) {
      const btn = el("button", "btn small", "Remove"); btn.type = "button";
      btn.onclick = () => removeDoc(d.source_url.replace("upload://", ""));
      act.appendChild(btn);
    } else if (d.saved_copy) {
      act.appendChild(link("Saved", d.saved_copy, "mini"));
    }
    tr.append(name, type, date, n, act);
    body.appendChild(tr);
  });
}

async function load(attempt = 1) {
  let docs, stats;
  try {
    [docs, stats] = await Promise.all([
      fetch("/api/documents").then(r => { if (!r.ok) throw new Error(r.status); return r.json(); }),
      fetch("/api/stats").then(r => { if (!r.ok) throw new Error(r.status); return r.json(); }),
    ]);
  } catch (err) {
    // The server may still be starting up (loading the index): retry a few times.
    $("docCount").textContent = attempt < 5 ? "Loading…" : "Could not load data. Is the server running?";
    if (attempt < 5) setTimeout(() => load(attempt + 1), 2000);
    return;
  }
  allDocs = docs.documents;
  $("nDocs").textContent = allDocs.length;
  $("nQs").textContent = stats.total_questions;
  $("avgMs").textContent = stats.total_questions ? (stats.avg_response_ms / 1000).toFixed(1) + " s" : "–";
  const chain = (stats.provider || "none").split(" -> ");
  $("aiName").textContent = chain[0] === "none" ? "None (passages only)" : chain[0].split(":")[0];
  $("aiName").title = stats.provider || "";
  renderDocs();

  const top = $("topSrc");
  top.textContent = "";
  if (!stats.top_sources.length) top.appendChild(el("li", "muted", "No questions yet"));
  stats.top_sources.forEach(s => {
    const item = el("li");
    const name = link(s.name, s.link, "rank-name"); name.title = s.name;
    item.append(name, el("span", "count", s.count + "×"));
    top.appendChild(item);
  });
}

function setMsg(text, cls) { $("upMsg").textContent = text; $("upMsg").className = "msgline " + (cls || ""); }

async function waitForRebuild() {
  setMsg("Re-indexing documents, please wait...");
  for (;;) {
    await new Promise(r => setTimeout(r, 1500));
    const s = await fetch("/api/status").then(r => r.json());
    if (!s.rebuilding) {
      if (s.last_rebuild_error) setMsg("Re-index failed: " + s.last_rebuild_error, "err");
      else setMsg("Done. The chatbot can now answer from the updated documents.", "ok");
      $("docCount").textContent = "Loading…";
load();
      return;
    }
  }
}

function pickFile(f) {
  chosenFile = f;
  const drop = $("drop");
  if (f) { drop.textContent = f.name; drop.classList.add("chosen"); }
  else { drop.innerHTML = 'Drag a PDF / DOCX / TXT here<br><span class="muted">or click to choose</span>'; drop.classList.remove("chosen"); }
}

async function upload() {
  if (!chosenFile) return setMsg("Choose a file first.", "err");
  const fd = new FormData();
  fd.append("file", chosenFile);
  fd.append("password", $("pw").value);
  const res = await fetch("/api/uploads", { method: "POST", body: fd });
  const data = await res.json();
  if (!res.ok) return setMsg(data.error, "err");
  pickFile(null);
  waitForRebuild();
}

async function removeDoc(name) {
  if (!confirm("Remove " + name + " from the chatbot?")) return;
  const res = await fetch("/api/uploads/" + encodeURIComponent(name), {
    method: "DELETE", headers: { "X-Admin-Password": $("pw").value },
  });
  const data = await res.json();
  if (!res.ok) return setMsg(data.error + (res.status === 403 ? " (enter it in the upload box)" : ""), "err");
  waitForRebuild();
}

const drop = $("drop");
drop.onclick = () => $("file").click();
drop.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file").click(); } };
$("file").onchange = e => pickFile(e.target.files[0]);
drop.ondragover = e => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = e => { e.preventDefault(); drop.classList.remove("over"); pickFile(e.dataTransfer.files[0]); };
$("upBtn").onclick = upload;
$("filter").oninput = renderDocs;
$("docCount").textContent = "Loading…";
load();
