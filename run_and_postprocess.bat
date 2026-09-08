@echo off
setlocal

cd /d "%~dp0"

echo === Starting main.py (Ctrl+C to stop recording) ===
uv run python src\main.py

echo === Recording finished. Running postprocess ===

set "LATEST_DIR="
for /f "delims=" %%d in ('dir /b /ad /o-d "data"') do (
    set "LATEST_DIR=%%d"
    goto :found_dir
)
:found_dir

if "%LATEST_DIR%"=="" (
    echo No session directory found under data\. Skipping postprocess.
    goto :end
)

set "DATA_DIR=data\%LATEST_DIR%"
set "OUTPUT_DIR=%DATA_DIR%\postprocess_output"

echo Target directory: %DATA_DIR%
uv run python src\postprocess.py "%DATA_DIR%" "%OUTPUT_DIR%"

:end
endlocal
pause
