@echo off
setlocal enabledelayedexpansion
chcp 936 >nul

echo ========================================
echo     Hugging Face Downloader Builder
echo ========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.9+ and check "Add to PATH".
    pause
    exit /b 1
)

echo Installing deps...
python -m pip install -U requests pyinstaller huggingface_hub

echo.
echo Building EXE...
echo.

if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

python -m PyInstaller --onefile --windowed --name HuggingFaceDownloader --clean --noconfirm main.py

if errorlevel 1 (
    echo.
    echo Build failed. Check messages above.
    pause
    exit /b 1
)

if exist "dist\HuggingFaceDownloader.exe" (
    echo.
    echo ========================================
    echo          SUCCESS
    echo ========================================
    echo dist\HuggingFaceDownloader.exe
    echo.
    explorer dist
) else (
    echo.
    echo ERROR: EXE not found in dist folder.
)

pause
