@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Через 5 секунд будет снимок экрана. Переключитесь на игру со столом!
timeout /t 5 >nul
python poker_tracker.py --check
start "" check.png
pause
