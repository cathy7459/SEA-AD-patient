@echo off
setlocal

set "PATIENT_ROOT=%~dp0"
for %%I in ("%PATIENT_ROOT%.") do set "PATIENT_ROOT=%%~fI"
if "%PATIENT_ROOT:~-1%"=="\" set "PATIENT_ROOT=%PATIENT_ROOT:~0,-1%"

set "REPO_ROOT=%PATIENT_ROOT%\..\GraphSAGE_pre"
for %%I in ("%REPO_ROOT%") do set "REPO_ROOT=%%~fI"

set "BAT_SCRIPT=%REPO_ROOT%\04_run_seaad_patient_association.bat"
if not exist "%BAT_SCRIPT%" (
  echo Launcher script not found: "%BAT_SCRIPT%"
  exit /b 1
)

"%BAT_SCRIPT%" -RepoRoot "%REPO_ROOT%" -PatientRoot "%PATIENT_ROOT%" %*
set "EXITCODE=%ERRORLEVEL%"
exit /b %EXITCODE%
