@echo off
REM Finds a real Python (Anaconda / Miniconda / python.org), not the
REM Windows-Store alias. Sets PY. Used by RUN_ALL.bat and INSTALL_PACKAGES.bat.
set PY=
for %%P in (
  "%USERPROFILE%\anaconda3\python.exe"
  "%USERPROFILE%\miniconda3\python.exe"
  "%LOCALAPPDATA%\anaconda3\python.exe"
  "%LOCALAPPDATA%\miniconda3\python.exe"
  "%ProgramData%\anaconda3\python.exe"
  "%ProgramData%\Anaconda3\python.exe"
  "%ProgramData%\miniconda3\python.exe"
  "C:\anaconda3\python.exe"
  "C:\Anaconda3\python.exe"
) do (
  if not defined PY if exist %%P set PY=%%~P
)
if not defined PY (
  for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
    if not defined PY if exist "%%D\python.exe" set PY=%%D\python.exe
  )
)
if not defined PY (
  where py >nul 2>nul && set PY=py
)
if not defined PY (
  echo *** Python not found. Open "Anaconda Prompt" from the Start menu,
  echo     cd to the repository folder and run the script again.
  exit /b 1
)
echo Using Python: %PY%
"%PY%" -c "import sys; print(sys.version)" || exit /b 1
