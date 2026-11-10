# Creates .venv-tts with Chatterbox (MIT) - the engine you may charge money for.
#
# Why a second environment: Chatterbox pins torch==2.6.0 and transformers==5.2.0, which
# fights the rest of DubTool's stack. So it lives on its own and DubTool shells out to it.
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_chatterbox.ps1
#
# Safe to re-run. Takes ~2-4 GB of downloads and a few minutes.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Run setup.ps1 first - it creates the main .venv this script copies the interpreter from."
}

$py = ".venv-tts\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "creating .venv-tts ..." -ForegroundColor Cyan
    & ".venv\Scripts\python.exe" -m venv .venv-tts
}

Write-Host "installing torch 2.6 (CPU) ..." -ForegroundColor Cyan
& $py -m pip install --upgrade pip --quiet
& $py -m pip install --quiet torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu

Write-Host "installing chatterbox-tts ..." -ForegroundColor Cyan
& $py -m pip install --quiet chatterbox-tts

Write-Host "`nverifying ..." -ForegroundColor Cyan
& $py -c "import chatterbox, torch; print('chatterbox ok, torch', torch.__version__)"

Write-Host @"

Done. Chatterbox is the default engine now (MIT weights - commercial use allowed).
Weights (~1 GB) download on the first dub into .cache\huggingface.

  .\dub.bat doctor
  .\dub.bat run "episode.mkv" --lang es
"@ -ForegroundColor Green
