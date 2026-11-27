@echo off
chcp 65001 >nul
cd /d "%~dp0"
set BB=
set /p BB="Большой блайнд стола (например 500 или 1000, Enter - определить самой): "
if "%BB%"=="" (python poker_tracker.py --hints) else (python poker_tracker.py --hints --bb %BB%)
pause
