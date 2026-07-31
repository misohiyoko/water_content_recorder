@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

echo === main.py を起動します (Ctrl+C で記録終了) ===
uv run python src\main.py

echo === 記録が終了しました。postprocess を実行します ===

set "LATEST_DIR="
for /f "delims=" %%d in ('dir /b /ad /o-d "data"') do (
    set "LATEST_DIR=%%d"
    goto :found_dir
)
:found_dir

if "%LATEST_DIR%"=="" (
    echo data フォルダにセッションディレクトリが見つかりませんでした。postprocess をスキップします。
    goto :end
)

set "DATA_DIR=data\%LATEST_DIR%"
set "OUTPUT_DIR=%DATA_DIR%\postprocess_output"

echo 対象ディレクトリ: %DATA_DIR%
uv run python src\postprocess.py "%DATA_DIR%" "%OUTPUT_DIR%"

:end
endlocal
pause
