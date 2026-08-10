@echo off
rem Convenience wrapper for building on this machine from a plain shell.
rem Usage: build.cmd [preset] [target]   (defaults: local-release, all)
rem Presets whose name contains "x86" build 32-bit.
setlocal
set PRESET=%1
if "%PRESET%"=="" set PRESET=local-release

rem Pick the matching MSVC toolchain for the preset's word size.
set TARGET_ARCH=x64
echo %PRESET% | findstr /C:"x86" >nul && set TARGET_ARCH=x86

set VS=C:\Program Files\Microsoft Visual Studio\18\Community
call "%VS%\Common7\Tools\VsDevCmd.bat" -arch=%TARGET_ARCH% -host_arch=x64 -no_logo || exit /b 1
set CMAKE=%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
set NINJA=%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe
cd /d %~dp0
"%CMAKE%" --preset %PRESET% -DCMAKE_MAKE_PROGRAM="%NINJA%" || exit /b 1
if "%2"=="" (
    "%CMAKE%" --build --preset %PRESET%
) else (
    "%CMAKE%" --build --preset %PRESET% --target %2
)
