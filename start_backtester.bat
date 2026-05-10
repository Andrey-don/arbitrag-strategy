@echo off
title MATS Backtester
cd /d "%~dp0"
if not exist "venv\Scripts\streamlit.exe" (
    echo [ОШИБКА] venv не найден. Запустите: python -m venv venv && venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)
echo Запускаю MATS Backtester на порту 8502...
venv\Scripts\streamlit.exe run src\monitoring\backtest.py --server.port 8502
pause
