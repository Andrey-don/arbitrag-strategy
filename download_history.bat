@echo off
title MATS — Загрузка истории
cd /d "%~dp0"
call venv\Scripts\activate
python src/monitoring/download_history.py
