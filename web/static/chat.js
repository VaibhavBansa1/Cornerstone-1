// IT Help Desk chat page.
// - answers are rendered with clickable citation numbers [1], [2, 3]
// - clicking a number (or a source card) opens the Sources panel on that source
// - each source can be opened at the original link or as our saved copy

const $ = id => document.getElementById(id);
const messages = $("messages"), form = $("composer"), input = $("q"), sendBtn = $("send");
const panel = $("panel"), panelBody = $("panelBody"), panelSub = $("panelSub"), scrim = $("scrim");
const answers = [];          // response data per bot message, so any message's sources can be reopened
const history = [];          // this chat's last few questions + answers, for follow-ups; memory only, gone on reload
let activeAnswer = null;

const TYPE_LABEL = { page: "Web page", pdf: "PDF", upload: "Uploaded doc", image: "Image",
                     video: "Video", audio: "Audio", document: "Document" };

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined && text !== null) e.textContent = text;
  return e;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// [2] / [1, 3] / [4-6] -> clickable chips
function citationsToHtml(html, answerIdx, valid) {
  return html.replace(/\[(\s*\d+(?:\s*[,–-]\s*\d+)*\s*)\]/g, (match, group) => {
    const nums = [];
    group.split(",").forEach(part => {
      const b = part.trim().split(/\s*[–-]\s*/);
      if (b.length === 2 && +b[1] > +b[0] && +b[1] - +b[0] < 20) {
        for (let i = +b[0]; i <= +b[1]; i++) nums.push(i);
      } else if (/^\d+$/.test(part.trim())) nums.push(+part.trim());
    });
    const ok = nums.filter(n => valid.has(n));
    if (!ok.length) return match;
    return ok.map(n => `<button type="button" class="cite" data-a="${answerIdx}" data-n="${n}" aria-label="Source ${n}">${n}</button>`).join("");
  });
}

// Small, safe Markdown subset: paragraphs, bullet lists, tables, **bold**, *italic*.
function renderMarkdown(text, answerIdx, valid) {
  const lines = escapeHtml(text.trim()).split(/\n/);
  let html = "", inList = false, table = null;
  const flushTable = () => {
    if (!table) return;
    const [head, ...body] = table;
    html += '<div class="table-wrap"><table><thead><tr>' + head.map(c => `<th>${c}</th>`).join("") + "</tr></thead><tbody>" +
            body.map(r => "<tr>" + r.map(c => `<td>${c}</td>`).join("") + "</tr>").join("") + "</tbody></table></div>";
    table = null;
  };
  for (const raw of lines) {
    const line = raw.trim();
    if (/^\|.*\|$/.test(line)) {                       // table row
      if (inList) { html += "</ul>"; inList = false; }
      if (/^\|[\s:|-]+\|$/.test(line)) continue;       // |---|---| separator
      (table = table || []).push(line.slice(1, -1).split("|").map(c => c.trim()));
      continue;
    }
    flushTable();
    const bullet = line.match(/^[*\-•]\s+(.*)/);
    if (bullet) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += `<li>${bullet[1]}</li>`;
      continue;
    }
    if (inList) { html += "</ul>"; inList = false; }
    if (line) html += `<p>${line}</p>`;
  }
  flushTable();
  if (inList) html += "</ul>";
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
             .replace(/(^|[\s(>])\*(?!\s)([^*<]+?)\*(?=[\s.,;:)<]|$)/g, "$1<em>$2</em>");
  return citationsToHtml(html, answerIdx, valid);
}

function addRow(who) {
  $("welcome")?.remove();
  const row = el("div", "msg " + who);
  if (who === "bot") row.appendChild(el("div", "avatar", "M"));
  const bubble = el("div", "bubble");
  row.appendChild(bubble);
  messages.appendChild(row);
  scrollDown();
  return bubble;
}

function scrollDown() { messages.scrollTop = messages.scrollHeight; }

function sourceMeta(s) {
  const bits = [TYPE_LABEL[s.source_type] || s.source_type];
  if (s.date && s.date !== "undated") bits.push(s.date);
  return bits.join(" · ");
}

function renderAnswer(bubble, data) {
  const idx = answers.push(data) - 1;
  const valid = new Set(data.sources.map(s => s.n));
  bubble.textContent = "";
  bubble.dataset.answer = idx;

  if (data.understood_as || data.rewritten) {
    const u = el("div", "understood");
    u.appendChild(el("span", "u-label", "Understood as"));
    const parts = [data.rewritten ? "“" + data.rewritten + "”" : "", data.understood_as].filter(Boolean);
    u.appendChild(document.createTextNode(" " + parts.join(" · ")));
    u.title = "How the question was read before searching. Add a batch (e.g. \"2024-28 batch\") or semester to change it.";
    bubble.appendChild(u);
  }

  if (data.generated) {
    const body = el("div", "answer");
    body.innerHTML = renderMarkdown(data.answer, idx, valid);
    bubble.appendChild(body);
  } else {
    bubble.appendChild(el("div", "note",
      "No AI model is connected right now, so here are the closest matching passages. Click one to open it."));
  }

  // Source cards under the answer: cited ones first.
  const cited = data.sources.filter(s => data.cited.includes(s.n));
  const shown = data.generated ? cited : data.sources.slice(0, 3);
  const row = el("div", "src-row");
  shown.forEach(s => {
    const card = el("button", "src-card");
    card.type = "button";
    card.dataset.a = idx; card.dataset.n = s.n;
    card.appendChild(el("span", "num", s.n));
    const txt = el("span", "src-txt");
    txt.appendChild(el("span", "src-title", s.title));
    txt.appendChild(el("span", "src-meta", sourceMeta(s)));
    card.appendChild(txt);
    row.appendChild(card);
  });
  const more = data.sources.length - shown.length;
  if (more > 0) {
    const all = el("button", "src-more", shown.length ? `+${more} more` : `${more} sources searched`);
    all.type = "button"; all.dataset.a = idx;
    row.appendChild(all);
  }
  bubble.appendChild(row);

  const foot = el("div", "foot");
  foot.appendChild(el("span", null,
    (data.generated ? "Answered by " + data.provider.split(":")[0] : "Passages only") +
    " · " + (data.response_ms / 1000).toFixed(1) + " s"));
  if (data.generated) {
    const copy = el("button", "linkish", "Copy");
    copy.type = "button";
    copy.addEventListener("click", () => {
      navigator.clipboard?.writeText(data.answer).then(() => { copy.textContent = "Copied"; setTimeout(() => copy.textContent = "Copy", 1500); });
    });
    foot.appendChild(copy);
  }
  bubble.appendChild(foot);

  showSources(idx, data.generated ? (cited[0] || {}).n : 1, false);
}

function linkButton(label, href, cls) {
  const a = el("a", "btn-link " + (cls || ""), label);
  a.href = href; a.target = "_blank"; a.rel = "noopener";
  return a;
}

function sourceCard(s, isCited, answerIdx) {
  const card = el("article", "panel-card" + (isCited ? " cited" : ""));
  card.id = `src-${answerIdx}-${s.n}`;
  const head = el("div", "pc-head");
  head.appendChild(el("span", "num", s.n));
  const t = el("div", "pc-title");
  t.appendChild(el("div", "pc-name", s.title));
  t.appendChild(el("div", "src-meta", sourceMeta(s) + (isCited ? " · cited" : "")));
  head.appendChild(t);
  card.appendChild(head);
  if (s.doc_title && s.doc_title !== s.title) card.appendChild(el("div", "pc-doc", s.doc_title));

  const passage = el("div", "pc-text collapsed", s.text || s.snippet);
  card.appendChild(passage);
  const toggle = el("button", "linkish", "Show full passage");
  toggle.type = "button";
  toggle.addEventListener("click", () => {
    passage.classList.toggle("collapsed");
    toggle.textContent = passage.classList.contains("collapsed") ? "Show full passage" : "Show less";
  });
  card.appendChild(toggle);

  const actions = el("div", "pc-actions");
  if (s.link) actions.appendChild(linkButton("Open original ↗", s.link, "primary"));
  if (s.saved_copy) actions.appendChild(linkButton("Saved copy", s.saved_copy));
  if (!s.link && !s.saved_copy) actions.appendChild(el("span", "src-meta", "No file available"));
  card.appendChild(actions);
  return card;
}

function showSources(answerIdx, focusN, openOnMobile = true) {
  const data = answers[answerIdx];
  if (!data) return;
  activeAnswer = answerIdx;
  document.querySelectorAll(".bubble.active").forEach(b => b.classList.remove("active"));
  document.querySelector(`.bubble[data-answer="${answerIdx}"]`)?.classList.add("active");

  panelSub.textContent = "For: “" + data.question + "”";
  panelBody.textContent = "";
  const cited = data.sources.filter(s => data.cited.includes(s.n));
  const others = data.sources.filter(s => !data.cited.includes(s.n));
  if (data.generated && cited.length) {
    panelBody.appendChild(el("div", "panel-label", "Used in the answer"));
    cited.forEach(s => panelBody.appendChild(sourceCard(s, true, answerIdx)));
  }
  if (others.length) {
    panelBody.appendChild(el("div", "panel-label",
      !data.generated ? "Closest passages" : cited.length ? "Also searched" : "Searched (none used in the answer)"));
    others.forEach(s => panelBody.appendChild(sourceCard(s, false, answerIdx)));
  }
  if (openOnMobile) openPanel();
  if (focusN) {
    const card = $(`src-${answerIdx}-${focusN}`);
    if (card) {
      card.scrollIntoView({ behavior: "smooth", block: "start" });
      card.classList.remove("flash"); void card.offsetWidth; card.classList.add("flash");
    }
  }
}

function openPanel() { document.body.classList.add("panel-open"); }
function closePanel() { document.body.classList.remove("panel-open"); }

// Live progress while answering: a short checklist that fills in as the
// server reports each step (see /api/query/stream).
function progressView(bubble) {
  bubble.textContent = "";
  const list = el("ol", "progress");
  bubble.appendChild(list);
  const started = Date.now();
  let current = null, timer = null;

  function add(text) {
    if (current) current.classList.replace("now", "done");
    current = el("li", "now");
    current.appendChild(el("span", "p-text", text));
    list.appendChild(current);
    scrollDown();
    return current;
  }
  function detail(li, text) { li.appendChild(el("span", "p-detail", text)); scrollDown(); }
  function stopTimer() { if (timer) clearInterval(timer); timer = null; }
  const first = add("Reading your question");   // shown at once, before the server replies

  return {
    step(ev) {
      if (ev.step === "rewritten") {
        detail(first, "Follow-up read as: " + ev.value);
      } else if (ev.step === "understood") {
        if (ev.value) detail(first, "Understood as: " + ev.value);
      } else if (ev.step === "searching") {
        add(`Searching ${Number(ev.value).toLocaleString("en-IN")} pieces of the MITS website`);
      } else if (ev.step === "found") {
        const li = add(ev.sources.length ? `Found ${ev.sources.length} matching sources` : "No matching sources found");
        // show the source names one by one, so it's clear what is being read
        ev.sources.slice(0, 4).forEach((t, i) => setTimeout(() => detail(li, "Reading: " + t), 150 * i));
        if (ev.sources.length > 4) setTimeout(() => detail(li, `+${ev.sources.length - 4} more`), 600);
      } else if (ev.step === "writing") {
        stopTimer();
        const name = ev.provider.split(":")[0];
        const li = add((ev.fallback ? "Previous AI was busy · now asking " : "Writing the answer with ") + name);
        const secs = el("span", "p-secs", "");
        li.appendChild(secs);
        timer = setInterval(() => { secs.textContent = ` · ${Math.round((Date.now() - started) / 1000)} s`; }, 1000);
      }
    },
    finish() { stopTimer(); },
  };
}

// Read the progress stream; returns the final answer object.
async function askStream(question, progress) {
  const res = await fetch("/api/query/stream", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, history: history.slice(-3) }),
  });
  if (!res.ok || !res.body) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.error || "Something went wrong.");
  }
  const reader = res.body.getReader(), decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, nl).trim();
      buffer = buffer.slice(nl + 1);
      if (!line) continue;
      const ev = JSON.parse(line);
      if (ev.type === "step") progress.step(ev);
      else if (ev.type === "answer") return ev;
      else if (ev.type === "error") throw new Error(ev.error);
    }
  }
  throw new Error("The connection closed before the answer arrived.");
}

async function ask(question) {
  addRow("me").textContent = question;
  const bubble = addRow("bot");
  const progress = progressView(bubble);
  sendBtn.disabled = true;
  try {
    const data = await askStream(question, progress);
    data.question = question;
    progress.finish();
    renderAnswer(bubble, data);
    if (data.generated) {
      history.push({ q: question, a: (data.answer || "").slice(0, 500) });
      if (history.length > 3) history.shift();
    }
  } catch (err) {
    progress.finish();
    bubble.textContent = "Sorry, " + err.message;
    bubble.classList.add("error");
  } finally {
    sendBtn.disabled = false;
    scrollDown();
    input.focus();
  }
}

// --- events -----------------------------------------------------------------
form.addEventListener("submit", e => {
  e.preventDefault();
  const q = input.value.trim();
  if (!q || sendBtn.disabled) return;
  input.value = ""; input.style.height = "";
  ask(q);
});
input.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
});
input.addEventListener("input", () => {
  input.style.height = "";
  input.style.height = Math.min(input.scrollHeight, 140) + "px";
});
messages.addEventListener("click", e => {
  const t = e.target.closest(".cite, .src-card, .src-more");
  if (t && t.dataset.a !== undefined) showSources(+t.dataset.a, t.dataset.n ? +t.dataset.n : null);
  const s = e.target.closest(".suggest button");
  if (s) ask(s.textContent);
});
$("closePanel").addEventListener("click", closePanel);
scrim.addEventListener("click", closePanel);
document.addEventListener("keydown", e => { if (e.key === "Escape") closePanel(); });
$("newChat").addEventListener("click", () => location.reload());

fetch("/api/status").then(r => r.json()).then(s => {
  const status = $("status");
  const banner = $("banner");
  if (!s.index_ready) {
    banner.className = "banner err";
    banner.textContent = "Document index not built yet. Run: python run_pipeline.py";
  }
  if (s.provider === "none") {
    status.textContent = "Passages only";
    status.className = "status off";
    status.title = "No AI model reachable: set OPENROUTER_API_KEY or GROQ_API_KEY, or run Ollama.";
  } else {
    const chain = s.provider.split(" -> ").map(p => p.split(":")[0]);
    status.textContent = "AI: " + chain[0] + (chain.length > 1 ? " (+" + (chain.length - 1) + " backup)" : "");
    status.className = "status on";
    status.title = "Answering with: " + s.provider;
  }
}).catch(() => { $("status").textContent = "Offline"; });
