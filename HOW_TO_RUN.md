# How to run MITS Help Desk (Windows)

No GPU needed. Internet is needed for the first setup and for the answers (the AI runs on Groq's servers).

## Start it

1. Put the project folder somewhere short, e.g. `C:\MITS_Help_Desk` (not inside OneDrive).
2. Open the folder, click the address bar, type `powershell`, press Enter.
3. Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_demo.ps1
```

4. The browser opens `http://localhost:5000` by itself. Ask a question.
5. To stop: close the PowerShell window (or press Ctrl+C).

**To start it again later (next day, after a restart, after `git pull`): run the same command.** The script checks what is already installed and skips it, so it only starts the chatbot (about a minute). Nothing is installed twice.

| You have | First run | Later runs |
|---|---|---|
| **The full folder** (zip with `data\` and `models\`) | 3-10 min: installs Python + libraries | about 1 min |
| **A copy from GitHub** (code only) | also crawls the MITS website and builds the index: 30-90 min, downloads the search models once | about 1 min |

**Before a demo, do the first run a day early, with good internet.**

**AI key:** the answers need a free key. Put it in the `.env` file: `GROQ_API_KEY=gsk_...` (get one at console.groq.com). The full folder already has one. The script makes `.env` from `.env.example` if it's missing. Without a key, the chat shows the matching passages instead of a written answer.

## Get updates

The full folder is already linked to GitHub. To update:

```powershell
git pull
powershell -ExecutionPolicy Bypass -File .\run_demo.ps1
```

The script reinstalls libraries only if they changed.

`git pull` brings **code only**. These stay as they are and are never touched by git:

| Not in GitHub | How to update it |
|---|---|
| `data\` (scraped pages, index) | Copy `data\index\` and `data\processed\` (about 70 MB) from the PC that rebuilt them. Or rebuild here: `.venv\Scripts\python.exe run_pipeline.py` |
| `models\` (search models) | Only if the model changes (not planned) |
| `.env` (keys, settings) | Edit it by hand |

A folder that came without `.git` can be linked to GitHub like this (needs Git; a private repo asks you to sign in):

```powershell
git init
git remote add origin https://github.com/VaibhavBansa1/MITS-Help-Desk.git
git fetch origin claude/modular-date-aware-rag
git checkout -f -t origin/claude/modular-date-aware-rag
```

## The same steps by hand

If the script fails, run these one at a time in PowerShell, inside the folder:

```powershell
# 1. uv installs Python and the libraries. Skip if "uv --version" already works.
powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"

# 2. Python 3.11 + libraries, CPU build (add --extra ocr if you will build the data)
uv sync --frozen --extra cpu --python 3.11

# 3. Settings for this window
$env:EMBEDDING_DEVICE = "cpu"
$env:PYTHONUTF8 = "1"
$env:HF_HOME = "$PWD\models"; $env:HF_HUB_OFFLINE = "1"   # only if the folder has models\

# 4. Only for a copy from GitHub: build the data
.venv\Scripts\python.exe run_pipeline.py

# 5. Start, then open http://localhost:5000
.venv\Scripts\python.exe -m web.app
```

## What is in the folder

| Item | What it is |
|---|---|
| `run_demo.ps1` | Setup + start, one command |
| `.env` | API keys and settings. Never shared on GitHub |
| `config.py`, `rag\`, `web\`, `ingestion\`, `scraper\`, `scripts\`, `run_pipeline.py` | The code |
| `pyproject.toml`, `uv.lock` | Exact list of libraries |
| `data\index\`, `data\processed\` | The search index the bot answers from (full folder only) |
| `data\raw\` | The scraped pages and PDFs; "open source" links show these (full folder only) |
| `models\` | The two search models (e5-small, MiniLM re-ranker), so nothing is downloaded (full folder only) |
| `.venv\` | Made by the first run; never copied, it only works on the computer that made it |
| `docs\`, `tests\`, `README.md` | Documentation and tests |

## Which AI answers

Groq (`gpt-oss-120b`) answers first. If it is busy or its daily limit is used up, the smaller Groq model answers, then OpenRouter. If none can (no internet), the chat shows the matching passages instead. The model that answered is shown under each answer.

## If something goes wrong

| Problem | Fix |
|---|---|
| "running scripts is disabled" | Use the full command above (`powershell -ExecutionPolicy Bypass -File ...`) |
| Installing the libraries failed | Check the internet; run the script again (it carries on) |
| Windows Firewall asks about Python | Either button is fine |
| Page already open / "address already in use" | It is already running; open `http://localhost:5000` |
| Answers say "matching passages" only | No internet, or no API key, or all keys used up for today |
| The first answer is slow | Normal: the search models load on the first question (~20-40 s on a laptop) |
| Admin page | `http://localhost:5000/admin` (password: `ADMIN_PASSWORD` in `.env`) |
