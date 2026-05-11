@echo off
title MATS Bot — SBER/SBERP
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo [ОШИБКА] venv не найден. Запустите: python -m venv venv ^&^& venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)
if not exist ".env" (
    echo [ОШИБКА] Файл .env не найден. Создайте .env и добавьте TINKOFF_TOKEN=...
    pause
    exit /b 1
)
echo Запускаю MATS Bot (DRY_RUN по умолчанию)...
echo Чтобы включить реальную торговлю: добавьте DRY_RUN=false в .env
echo.
venv\Scripts\python.exe -m src.bot
pause
