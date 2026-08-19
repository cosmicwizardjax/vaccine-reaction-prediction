@echo off
title Vaccine Reaction Prediction - Setup
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Setup complete.
echo Start the application with: run_app.bat
pause
