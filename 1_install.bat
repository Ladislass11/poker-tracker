@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Устанавливаю библиотеки...
python -m pip install -r requirements.txt
echo.
echo Готово. Теперь запустите 2_check.bat
pause
