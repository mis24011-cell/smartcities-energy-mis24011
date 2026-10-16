@echo off
REM Installs the packages of requirements.txt into the Anaconda/Python found by find_python.bat.
REM Run once, from the repository root:   v2\INSTALL_PACKAGES.bat
call v2\find_python.bat || exit /b 1
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 ( echo *** pip failed & exit /b 1 )
echo PACKAGES OK - now run  v2\RUN_ALL.bat
