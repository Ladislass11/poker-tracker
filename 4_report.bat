@echo off
chcp 65001 >nul
cd /d "%~dp0"
python poker_tracker.py --report
pause
