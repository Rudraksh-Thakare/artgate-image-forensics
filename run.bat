@echo off
title ArtGate - Universal AI Image Forensics
echo ============================================================
echo   ArtGate: Universal AI Image Forensics & Detector
echo ============================================================
echo.
echo [*] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to your system PATH!
    echo Please install Python 3.10 or newer from https://www.python.org/
    echo and make sure to check "Add Python to PATH" during installation.
    pause
    exit /b
)

echo [*] Checking required dependencies...
python -c "import torch, torchvision, transformers, flask, cv2, pywt, sklearn, joblib" >nul 2>&1
if %errorlevel% neq 0 (
    echo [*] Installing missing dependencies from requirements.txt...
    pip install -r requirements.txt
)

echo.
echo [*] Starting ArtGate Universal AI Forensics Server...
echo [*] Opening Web Workbench at http://localhost:5000 ...
echo.
timeout /t 2 >nul
start http://localhost:5000
python server.py
pause
