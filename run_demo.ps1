<#
  MITS Help Desk - set up (first time) and start the chatbot on a Windows PC or laptop.

  Run it from this folder:
      powershell -ExecutionPolicy Bypass -File .\run_demo.ps1

  First run: needs internet, about 3-10 minutes (installs Python + libraries, ~1 GB, into .venv here).
  Later runs: about a minute. After "git pull" it reinstalls the libraries only if they changed.
  Runs on the CPU, no GPU needed. Stop it with Ctrl+C or by closing the window.
  See HOW_TO_RUN.md for the same steps by hand.
#>
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Url = "http://localhost:5000"

function Step($text) { Write-Host "`n== $text" -ForegroundColor Cyan }
function Stop-With($text) { Write-Host "`nPROBLEM: $text" -ForegroundColor Red; Read-Host "Press Enter to close"; exit 1 }

# ---- Settings --------------------------------------------------------------------------
if (-not $env:EMBEDDING_DEVICE) { $env:EMBEDDING_DEVICE = "cpu" }   # works on every laptop
$env:UV_NO_SYNC = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
# The search models shipped in the "models" folder (full copy only): use them, never download.
if (Test-Path "models\hub") {
    $env:HF_HOME = Join-Path $PSScriptRoot "models"
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
}
$hasIndex = (Test-Path "data\index\faiss.index") -and (Test-Path "data\processed\chunks.jsonl")

# ---- 1. Check the folder ---------------------------------------------------------------
Step "1/4 Checking files"
foreach ($f in "config.py", "web\app.py", "pyproject.toml", "uv.lock") {
    if (-not (Test-Path $f)) { Stop-With "missing $f - run this script from the project folder." }
}
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Made a .env file from .env.example. Put your GROQ_API_KEY in it (free: console.groq.com)." -ForegroundColor Yellow
}
if (-not (Select-String -Path ".env" -Pattern "^\s*(GROQ|OPENROUTER)_API_KEY\s*=\s*\S+" -Quiet)) {
    Write-Host "Warning: no GROQ_API_KEY / OPENROUTER_API_KEY in .env - the chat will show matching passages instead of AI answers." -ForegroundColor Yellow
}
if (-not $hasIndex) { Write-Host "No scraped data yet (data\index is empty). It will be built after the install." -ForegroundColor Yellow }
Write-Host "OK"

# ---- 2. uv (installs Python and the libraries) -------------------------------------------
Step "2/4 uv"
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv (needs internet)..."
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { Stop-With "uv did not install. Check the internet and run this script again." }
}
uv --version

# ---- 3. Python 3.11 + libraries, CPU build (again only when uv.lock changes) ---------------
Step "3/4 Python libraries"
$python = ".venv\Scripts\python.exe"
$extras = @("--extra", "cpu")
if (-not $hasIndex) { $extras += @("--extra", "ocr") }      # building the data needs OCR for scanned PDFs
$want = "$((Get-FileHash uv.lock).Hash) $($extras -join ' ')"
$have = if (Test-Path ".venv\demo_ready.txt") { (Get-Content ".venv\demo_ready.txt" -Raw).Trim() } else { "" }
if ($want -ne $have -or -not (Test-Path $python)) {
    Write-Host "Installing Python 3.11 and the libraries (first run or libraries changed; 3-10 minutes)..."
    uv sync --frozen @extras --python 3.11
    if ($LASTEXITCODE -ne 0) { Stop-With "installing the libraries failed. Check the internet and run this script again." }
    & $python -c "import torch, faiss, sentence_transformers, flask; print('torch', torch.__version__, '| GPU used:', torch.cuda.is_available())"
    if ($LASTEXITCODE -ne 0) { Stop-With "the libraries installed but do not load." }
    Set-Content ".venv\demo_ready.txt" $want
}
Write-Host "OK"

# ---- Build the data if this is a fresh copy from GitHub ----------------------------------------
if (-not $hasIndex) {
    Step "Building the data: crawling the MITS website and reading every PDF (30-90 minutes on a laptop)"
    & $python run_pipeline.py
    if ($LASTEXITCODE -ne 0) { Stop-With "building the data failed - see the messages above, then run this script again." }
}

# ---- 4. Start the chatbot ----------------------------------------------------------------------
Step "4/4 Starting the chatbot"
$running = $false
try { Invoke-WebRequest "$Url/api/status" -UseBasicParsing -TimeoutSec 2 | Out-Null; $running = $true } catch { }
if ($running) {
    Write-Host "It is already running - opening $Url"
    Start-Process $Url
    exit 0
}
# Open the browser as soon as the page answers.
Start-Job -ArgumentList $Url -ScriptBlock {
    param($u)
    for ($i = 0; $i -lt 180; $i++) {
        try { Invoke-WebRequest $u -UseBasicParsing -TimeoutSec 2 | Out-Null; Start-Process $u; return } catch { Start-Sleep 1 }
    }
} | Out-Null
Write-Host "Chat:  $Url"
Write-Host "Admin: $Url/admin"
Write-Host "The first answer takes a little longer while the search models load. Close this window to stop."
Write-Host "If Windows Firewall asks about Python, either button is fine (only this computer needs it)."
& $python -m web.app
