@echo off
rem Opens the DubTool window. Double-click this, or the shortcut in testing\.
setlocal
set "EXE=%~dp0dist\DubTool.exe"
if exist "%EXE%" (
    start "" "%EXE%"
    exit /b 0
)
set "PYW=%~dp0.venv\Scripts\pythonw.exe"
if exist "%PYW%" (
    start "" "%PYW%" "%~dp0launch_gui.py"
    exit /b 0
)
echo DubTool.exe is not built yet and there is no .venv next to this file.
echo   powershell -ExecutionPolicy Bypass -File scripts\build_exe.ps1
pause
