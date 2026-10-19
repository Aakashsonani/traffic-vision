@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.ps1 first to install Python dependencies and models.
  pause
  exit /b 1
)
echo Traffic Vision will run at http://127.0.0.1:8765
echo Keep this window open. Press Ctrl+C to stop.
start "" "http://127.0.0.1:8765"
".venv\Scripts\python.exe" -m traffic_vision
if errorlevel 1 pause
