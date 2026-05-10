@echo off
title MATS Dashboard
cd /d "%~dp0"
if not exist "venv\Scripts\streamlit.exe" (
    echo [ОШИБКА] venv не найден. Запустите: python -m venv venv && venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)
echo Запускаю MATS Dashboard на порту 8501...
venv\Scripts\streamlit.exe run src\monitoring\dashboard.py --server.address 0.0.0.0 --server.port 8501
pause
