# Builds DubTool.exe - one file, double-clickable, no Python needed.
#
#   powershell -ExecutionPolicy Bypass -File scripts\build_exe.ps1
#
# The bundle holds the window, the live/mode engine (piper), the transcript model
# (faster-whisper tiny/small) and ffmpeg. The cloning engine (chatterbox) drags in torch,
# which would triple the size, so it is deliberately left out: the app notices an installed
# `.venv-tts` and uses it when present.

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { throw "no venv at $py - run install.ps1 first" }

Write-Host "cleaning..." -ForegroundColor Cyan
Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue

$exclude = @(
    'torch', 'torchaudio', 'torchvision', 'transformers', 'chatterbox', 'demucs',
    'pyannote', 'speechbrain', 'sklearn', 'scipy', 'pandas', 'matplotlib', 'IPython',
    'pytest', 'PIL', 'cv2', 'tensorboard', 'jupyter', 'notebook', 'tkinter.test',
    'test', 'unittest', 'pydoc_data'
) | ForEach-Object { @('--exclude-module', $_) }

$hidden = @(
    'dub', 'dubkit.gui', 'dubkit.live', 'dubkit.stream', 'dubkit.voice', 'dubkit.speech',
    'dubkit.pipeline', 'dubkit.batch', 'dubkit.ff'
) | ForEach-Object { @('--hidden-import', $_) }

$collect = @('imageio_ffmpeg', 'piper', 'faster_whisper', 'ctranslate2', 'av',
             'soundfile', 'onnxruntime') | ForEach-Object { @('--collect-all', $_) }

Write-Host "building (a few minutes)..." -ForegroundColor Cyan
& $py -m PyInstaller --noconfirm --clean --onefile --windowed `
    --name DubTool `
    --icon assets\dubtool.ico `
    --add-data "dubkit\player.html;dubkit" `
    --distpath dist --workpath build `
    @exclude @hidden @collect `
    launch_gui.py
if ($LASTEXITCODE -ne 0) { throw "pyinstaller failed" }

$exe = Join-Path $root 'dist\DubTool.exe'
$mb = [math]::Round((Get-Item $exe).Length / 1MB, 1)
Write-Host "`nbuilt $exe  ($mb MB)" -ForegroundColor Green
Write-Host "double-click it, or copy it anywhere - it needs nothing installed."
