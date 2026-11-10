@echo off
rem Convenience wrapper: dub.bat run "D:\Shows\S01E01.mkv" --lang es
"%~dp0.venv\Scripts\python.exe" "%~dp0dub.py" %*
