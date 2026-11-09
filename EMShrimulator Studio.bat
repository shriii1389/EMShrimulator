@echo off
REM Double-click launcher for EMShrimulator Studio.
REM Starts the desktop application from this checkout. Everything else lives in gems.ps1.
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0gems.ps1" run
if errorlevel 1 (
  echo.
  echo The app did not start. Run this for a full diagnosis:
  echo     powershell -ExecutionPolicy Bypass -File "%~dp0gems.ps1" doctor
  echo.
  pause
)
