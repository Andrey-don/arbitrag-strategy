@echo off
title MATS Backtester
cd /d "%~dp0"
call venv\Scripts\activate
streamlit run src/monitoring/backtest.py --server.port 8502
pause
