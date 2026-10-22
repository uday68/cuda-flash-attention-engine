@echo off
setlocal

:: Locate MSVC host compiler
set MSVC_BIN=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64
set NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe

if "%~1"=="" (
    echo Usage: build.bat ^<source_file.cu^> [output_executable.exe]
    echo Example: build.bat tests\test_env.cu bin\test_env.exe
    exit /b 1
)

set SRC_FILE=%~1

if "%~2"=="" (
    set OUT_FILE=%~dpn1.exe
) else (
    set OUT_FILE=%~2
)

:: Create output directory if it doesn't exist
if not exist "bin" mkdir bin

echo Compiling %SRC_FILE% -> %OUT_FILE% for sm_86 (Ampere RTX 3050)...
"%NVCC%" -ccbin "%MSVC_BIN%" -O3 -std=c++17 -arch=sm_86 -I"./include" "%SRC_FILE%" -o "%OUT_FILE%"

if %ERRORLEVEL% equ 0 (
    echo [SUCCESS] Built: %OUT_FILE%
) else (
    echo [FAILED] Compilation failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

