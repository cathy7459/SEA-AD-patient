@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PATIENT_ROOT=%%~fI"
for %%I in ("%PATIENT_ROOT%\..\GraphSAGE_pre") do set "REPO_ROOT=%%~fI"

set "BAT_SCRIPT=%REPO_ROOT%\04_run_seaad_patient_association.bat"
if not exist "%BAT_SCRIPT%" (
  echo Launcher script not found: "%BAT_SCRIPT%"
  exit /b 1
)

"%BAT_SCRIPT%" -RepoRoot "%REPO_ROOT%" -PatientRoot "%PATIENT_ROOT%" %*
set "EXITCODE=%ERRORLEVEL%"
exit /b %EXITCODE%
