@echo off
title BioSentinel v2.3 — AI Health Monitor
echo.
echo  ============================================
echo   BioSentinel v2.3 — AI Health Monitor
echo   by Liveupx Pvt. Ltd.
echo  ============================================
echo.
python run.py
if errorlevel 1 (
    echo.
    echo  ERROR: Python not found.
    echo  Download from: https://python.org/downloads
    echo  Make sure to check "Add Python to PATH" during install.
)
pause
