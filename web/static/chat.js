const messages = document.getElementById("messages");
const form = document.getElementById("composer");
const input = document.getElementById("q");
const sendBtn = document.getElementById("send");
const banner = document.getElementById("banner");

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function addMessage(who) {
  document.getElementById("welcome")?.remove();
  const row = el("div", "msg " + who);
  const bubble = el("div", "bubble");
  row.appendChild(bubble);
  messages.appendChild(row);
  messages.scrollTop = messages.scrollHeight;
  return bubble;
}

function sourceChip(s) {
  const chip = el(s.link ? "a" : "span", "src", "[" + s.n + "] " + s.title);
  if (s.link) { chip.href = s.link; chip.target = "_blank"; chip.rel = "noopener"; }
  return chip;
}

function renderAnswer(bubble, data) {
  bubble.textContent = "";
  if (data.generated) {
    bubble.appendChild(document.createTextNode(data.answer));
    const cited = data.sources.filter(s => data.cited.includes(s.n));
    const box = el("div", "sources");
    box.appendChild(el("span", "meta", "Source:"));
    cited.forEach(s => box.appendChild(sourceChip(s)));
    bubble.appendChild(box);
  } else {
    bubble.appendChild(el("div", "note", "No AI model is connected right now, so here are the closest matching passages:"));
    data.sources.slice(0, 3).forEach(s => {
      const p = el("div", "passage");
      p.appendChild(el("b", null, "[" + s.n + "] " + s.title));
      p.appendChild(document.createTextNode(s.snippet + (s.snippet.length >= 400 ? "..." : "")));
      bubble.appendChild(p);
    });
    const box = el("div", "sources");
    data.sources.slice(0, 3).forEach(s => box.appendChild(sourceChip(s)));
    bubble.appendChild(box);
  }
  bubble.appendChild(el("div", "meta", (data.response_ms / 1000).toFixed(1) + " s"));
}

async function ask(question) {
  addMessage("me").textContent = question;
  const bubble = addMessage("bot");
  bubble.innerHTML = '<span class="typing"><span></span><span></span><span></span></span>';
  sendBtn.disabled = true;
  try {
    const res = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Something went wrong.");
    renderAnswer(bubble, data);
  } catch (err) {
    bubble.textContent = "Sorry, " + err.message;
  } finally {
    sendBtn.disabled = false;
    messages.scrollTop = messages.scrollHeight;
    input.focus();
  }
}

form.addEventListener("submit", e => {
  e.preventDefault();
  const q = input.value.trim();
  if (!q) return;
  input.value = "";
  ask(q);
});

document.querySelectorAll(".suggest button").forEach(b =>
  b.addEventListener("click", () => ask(b.textContent)));

fetch("/api/status").then(r => r.json()).then(s => {
  if (!s.index_ready) {
    banner.className = "banner err";
    banner.textContent = "Document index not built yet. Run: python run_pipeline.py";
  } else if (s.provider === "none") {
    banner.className = "banner";
    banner.textContent = "No AI model configured. Showing matching passages only (set GEMINI_API_KEY or run Ollama for full answers).";
  }
}).catch(() => {});
