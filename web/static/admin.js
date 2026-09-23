const $ = id => document.getElementById(id);
let allDocs = [];
let chosenFile = null;

function li(left, right) {
  const item = document.createElement("li");
  const a = document.createElement("span"); a.textContent = left;
  const b = document.createElement("span"); b.textContent = right; b.style.color = "#667085";
  item.append(a, b);
  return item;
}

function renderDocs() {
  const f = $("filter").value.toLowerCase();
  const body = $("docs");
  body.textContent = "";
  const shown = allDocs.filter(d => (d.title + " " + d.source_type).toLowerCase().includes(f));
  if (!shown.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td"); td.colSpan = 4;
    td.textContent = allDocs.length ? "No match." : "Nothing indexed yet. Run python run_pipeline.py or upload a file.";
    tr.appendChild(td); body.appendChild(tr);
    return;
  }
  shown.forEach(d => {
    const tr = document.createElement("tr");
    const name = document.createElement("td"); name.className = "title";
    if (d.source_url.startsWith("http")) {
      const a = document.createElement("a"); a.href = d.source_url; a.target = "_blank"; a.textContent = d.title;
      name.appendChild(a);
    } else name.textContent = d.title;
    const type = document.createElement("td");
    const tag = document.createElement("span"); tag.className = "tag"; tag.textContent = d.source_type; type.appendChild(tag);
    const n = document.createElement("td"); n.textContent = d.chunks;
    const act = document.createElement("td");
    if (d.deletable) {
      const btn = document.createElement("button"); btn.className = "btn small"; btn.textContent = "Remove";
      btn.onclick = () => removeDoc(d.source_url.replace("upload://", ""));
      act.appendChild(btn);
    }
    tr.append(name, type, n, act);
    body.appendChild(tr);
  });
}

async function load() {
  const [docs, stats] = await Promise.all([
    fetch("/api/documents").then(r => r.json()),
    fetch("/api/stats").then(r => r.json()),
  ]);
  allDocs = docs.documents;
  $("nDocs").textContent = allDocs.length;
  $("nQs").textContent = stats.total_questions;
  $("avgMs").textContent = stats.total_questions ? (stats.avg_response_ms / 1000).toFixed(1) + " s" : "-";
  renderDocs();

  $("topSrc").textContent = "";
  if (!stats.top_sources.length) $("topSrc").appendChild(li("No questions yet", ""));
  stats.top_sources.forEach(s => $("topSrc").appendChild(li(s.name, s.count + "x")));

  $("recent").textContent = "";
  if (!stats.recent_questions.length) $("recent").appendChild(li("No questions yet", ""));
  stats.recent_questions.forEach(q =>
    $("recent").appendChild(li(q.question, new Date(q.asked_at * 1000).toLocaleString())));
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
      load();
      return;
    }
  }
}

function pickFile(f) {
  chosenFile = f;
  $("drop").textContent = f ? f.name : "Drag a PDF / DOCX / TXT here or click to choose";
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
$("file").onchange = e => pickFile(e.target.files[0]);
drop.ondragover = e => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = e => { e.preventDefault(); drop.classList.remove("over"); pickFile(e.dataTransfer.files[0]); };
$("upBtn").onclick = upload;
$("filter").oninput = renderDocs;
load();
