@echo off
REM start_turboquant.bat - TurboQuant server baslatir
REM Kullanim: start_turboquant.bat [model_adi] [port]
REM
REM Ornek:
REM   start_turboquant.bat
REM   start_turboquant.bat liquid/lfm2-24b-a2b 8000

set MODEL=%1
if "%MODEL%"=="" set MODEL=liquid/lfm2-24b-a2b

set PORT=%2
if "%PORT%"=="" set PORT=8000

echo ========================================
echo   TurboQuant Server
echo   Model: %MODEL%
echo   Port: %PORT%
echo   Bits: 4 (4x bellek tasarrufu)
echo ========================================
echo.

REM turboquant yuklu mu kontrol et
pip show turboquant >nul 2>&1
if errorlevel 1 (
    echo turboquant yuklu degil, yukleniyor...
    pip install turboquant
)

echo Server baslatiliyor...
echo LLM modu icin baska bir terminalde:
echo   python autoresearch_loop.py --mode llm --use-turboquant
echo.

turboquant-server --model %MODEL% --bits 4 --port %PORT%
