@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

echo ============================================================
echo  TraFix - Python 3.12 + CARLA Ortam Kurulumu
echo ============================================================
echo.

set "CARLA_ROOT=C:\Users\Furkan\Downloads\CARLA_0.9.16"
set "CARLA_WHEEL=%CARLA_ROOT%\PythonAPI\carla\dist\carla-0.9.16-cp312-cp312-win_amd64.whl"
set "VENV_DIR=%~dp0.venv"
set "REQS=%~dp0requirements.txt"

:: ── 1. Python 3.12 kontrolu ──────────────────────────────────────────────────
echo [1/5] Python 3.12 kontrol ediliyor...
py -3.12 --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [HATA] Python 3.12 bulunamadi!
    echo.
    echo Lutfen Python 3.12'yi kurun:
    echo   Yontem A ^(Winget - otomatik^):
    echo     winget install -e --id Python.Python.3.12
    echo.
    echo   Yontem B ^(Manuel^):
    echo     https://www.python.org/downloads/release/python-3129/
    echo     "Windows installer (64-bit)" indirin ve kurun
    echo     Kurulum sirasinda "Add Python to PATH" secenegini isaretle!
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('py -3.12 --version 2^>^&1') do set PY312_VER=%%i
echo [OK] %PY312_VER% bulundu.

:: ── 2. CARLA wheel kontrolu ──────────────────────────────────────────────────
echo.
echo [2/5] CARLA wheel dosyasi kontrol ediliyor...
if not exist "%CARLA_WHEEL%" (
    echo [HATA] CARLA wheel bulunamadi: %CARLA_WHEEL%
    echo.
    echo CARLA_ROOT degiskenini guncelle: kurulum_python312.bat dosyasini
    echo Not Defteri ile ac ve satir 9'daki CARLA_ROOT yolunu duzenle.
    pause
    exit /b 1
)
echo [OK] Wheel bulundu: %CARLA_WHEEL%

:: ── 3. .venv olustur (Python 3.12) ──────────────────────────────────────────
echo.
echo [3/5] .venv olusturuluyor (Python 3.12)...
if exist "%VENV_DIR%" (
    echo [!] Mevcut .venv siliniyor...
    rmdir /s /q "%VENV_DIR%"
)
py -3.12 -m venv "%VENV_DIR%"
if %ERRORLEVEL% NEQ 0 (
    echo [HATA] venv olusturulamadi!
    pause
    exit /b 1
)
echo [OK] .venv olusturuldu.

:: ── 4. CARLA wheel kurulumu ──────────────────────────────────────────────────
echo.
echo [4/5] CARLA Python API kuruluyor...
"%VENV_DIR%\Scripts\pip.exe" install --no-deps "%CARLA_WHEEL%"
if %ERRORLEVEL% NEQ 0 (
    echo [HATA] CARLA kurulamadi!
    pause
    exit /b 1
)
echo [OK] CARLA kuruldu.

:: ── 5. Proje gereksinimleri ──────────────────────────────────────────────────
echo.
echo [5/5] Proje gereksinimleri kuruluyor...
if exist "%REQS%" (
    "%VENV_DIR%\Scripts\pip.exe" install -r "%REQS%"
) else (
    echo [UYARI] requirements.txt bulunamadi, atlanıyor.
)

:: pip upgrade
"%VENV_DIR%\Scripts\pip.exe" install --upgrade pip >nul 2>&1

:: ── Dogrulama ────────────────────────────────────────────────────────────────
echo.
echo ────────────────────────────────────────────────────────────
echo  Kurulum tamamlandi! CARLA import testi:
echo ────────────────────────────────────────────────────────────
"%VENV_DIR%\Scripts\python.exe" -c "import carla; print('[OK] carla modulu yuklendi.')" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [HATA] carla import basarisiz.
) else (
    echo.
    echo ============================================================
    echo  BASARILI! Artik calistirabilirsini:
    echo    python baslat.py --sumo-gui
    echo ============================================================
)

echo.
pause
