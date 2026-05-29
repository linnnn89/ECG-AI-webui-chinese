@echo off
title ECG AI Service - FastAPI
cd /d "%~dp0"

set "ECG_DIGITIZER_BACKEND=ahus"
set "PYTHONDONTWRITEBYTECODE=1"

echo ======================================================
echo   ECG AI backend window
echo   Keep this window open. Closing it stops the service.
echo   Image digitizer backend: %ECG_DIGITIZER_BACKEND%
echo ======================================================
echo.

echo [0/4] Checking whether 127.0.0.1:8000 is already in use...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p=(Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty OwningProcess); if ($p) { Write-Host ('Port 8000 is already used by PID=' + $p); exit 1 }"
if errorlevel 1 (
  echo.
  echo No new service was started. If the WebUI still opens, it is using an old backend.
  echo Close the old ECG AI backend window or stop the PID shown above, then run START.BAT again.
  pause
  exit /b 1
)

echo [1/4] Activating virtual environment...
if not exist ".venv\Scripts\activate.bat" (
  echo Missing .venv\Scripts\activate.bat. Please check the project virtual environment.
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"

echo [2/4] Opening WebUI after a short delay...
start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 3; Start-Process 'http://127.0.0.1:8000/ui/'"

echo [3/4] Starting FastAPI service on 127.0.0.1:8000...
echo ======================================================
echo   ECG inference backend is running.
echo   Hardware target: NVIDIA RTX 50-series, CUDA 12.x+
echo   WebUI: http://127.0.0.1:8000/ui/
echo ======================================================

python -m uvicorn code.service.app:APP --host 127.0.0.1 --port 8000

echo.
echo Service stopped.
pause
