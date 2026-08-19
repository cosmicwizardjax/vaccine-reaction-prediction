@echo off
title Vaccine Reaction Prediction
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo Python environment not found.
    echo Please double-click setup.bat first.
    echo.
    pause
    exit /b 1
)

echo Starting Vaccine Reaction Prediction...
echo The application will open in your browser.
echo Close this window only when you want to stop the local application.
echo.

".venv\Scripts\python.exe" -m streamlit run app.py --server.headless true --server.address 127.0.0.1
