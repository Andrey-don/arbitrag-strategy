@echo off
title MATS — Загрузка истории
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo [ОШИБКА] venv не найден. Запустите: python -m venv venv && venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)
echo Запускаю загрузку истории...
venv\Scripts\python.exe src\monitoring\download_history.py
pause
