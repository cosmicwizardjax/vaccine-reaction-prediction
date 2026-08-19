@echo off
title Vaccine Reaction Prediction - Local Network
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Please run setup.bat first.
    pause
    exit /b 1
)

echo ============================================================
echo Vaccine Reaction Prediction - Local Network Mode
echo ============================================================
echo.
echo This laptop must remain ON for other devices to use this link.
echo Connect the other device to the same Wi-Fi/network.
echo.
echo Your local network addresses are:
ipconfig | findstr /R /C:"IPv4 Address"
echo.
echo The Streamlit port is 8501.
echo Example: http://YOUR-IP:8501
echo.
echo Starting application...
echo Keep this window open while other devices are using it.
echo.

".venv\Scripts\python.exe" -m streamlit run app.py --server.headless true --server.address 0.0.0.0
