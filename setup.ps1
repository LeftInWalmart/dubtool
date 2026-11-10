# Creates .venv with Python 3.11 (torch has no 3.14 wheels) and installs deps.
#   powershell -ExecutionPolicy Bypass -File setup.ps1
#   powershell -ExecutionPolicy Bypass -File setup.ps1 -Cuda
#   powershell -ExecutionPolicy Bypass -File setup.ps1 -Extras
param([switch]$Cuda, [switch]$Extras, [string]$Python = "3.11")

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $here ".venv\Scripts\python.exe"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "installing uv ..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

if (-not (Test-Path $py)) {
    Write-Host "creating venv (python $Python) ..."
    uv venv (Join-Path $here ".venv") --python $Python
}

$index = if ($Cuda) { "https://download.pytorch.org/whl/cu124" } else { "https://download.pytorch.org/whl/cpu" }
Write-Host "installing torch ($index) ..."
uv pip install --python $py "torch>=2.4,<2.9" "torchaudio>=2.4,<2.9" --index-url $index

Write-Host "installing the rest ..."
uv pip install --python $py -r (Join-Path $here "requirements.txt")
if ($Extras) {
    uv pip install --python $py -r (Join-Path $here "requirements-extras.txt")
}

& $py (Join-Path $here "dub.py") doctor
