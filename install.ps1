# One command to go from a fresh clone to a working dub. Idempotent - safe to re-run.
#
#   powershell -ExecutionPolicy Bypass -File install.ps1
#   powershell -ExecutionPolicy Bypass -File install.ps1 -Cuda      # NVIDIA GPU
#   powershell -ExecutionPolicy Bypass -File install.ps1 -NoVoices  # skip the 2 GB cloning engine
#
# It installs three things: the app (torch + whisper + demucs), piper (fast, MIT, used by
# `dub.py live`), and Chatterbox (slow, MIT, clones the actor - the good-sounding default).
param([switch]$Cuda, [switch]$NoVoices, [switch]$Extras)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $here ".venv\Scripts\python.exe"
$dub = Join-Path $here "dub.py"

function Step($text) { Write-Host "`n== $text" -ForegroundColor Cyan }

Step "1/4  base environment"
& (Join-Path $here "setup.ps1") -Cuda:$Cuda -Extras:$Extras
if ($LASTEXITCODE -ne 0) { throw "setup.ps1 failed" }

Step "2/4  fast voice engine (piper, MIT)"
& $py -m pip install --quiet --disable-pip-version-check piper-tts yt-dlp
if ($LASTEXITCODE -ne 0) { throw "piper install failed" }

if (-not $NoVoices) {
    Step "3/4  cloning engine (chatterbox, ~2 GB)"
    & (Join-Path $here "scripts\setup_chatterbox.ps1")
} else {
    Step "3/4  cloning engine skipped (-NoVoices)"
}

Step "4/4  check"
& $py $dub doctor

Write-Host @"

Ready. Try it without hunting for footage - the 100 second demo clip ships with the repo:

  .\.venv\Scripts\python.exe dub.py run   demo\excerpt.mp4 --lang es
  .\.venv\Scripts\python.exe dub.py live  demo\excerpt.mp4 --lang es
  .\.venv\Scripts\python.exe dub.py batch demo\lines.example.csv --voices demo\cast --lang es

`dub.bat` does the ".\.venv\Scripts\python.exe" part for you:

  dub.bat live demo\excerpt.mp4 --lang es
"@ -ForegroundColor Gray
