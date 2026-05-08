@echo off
title MATS Dashboard
cd /d "%~dp0"
call venv\Scripts\activate
streamlit run src/monitoring/dashboard.py
pause
