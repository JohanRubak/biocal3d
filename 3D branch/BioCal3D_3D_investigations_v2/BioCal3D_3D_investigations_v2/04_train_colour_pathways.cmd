@echo off
call "%~dp0_run.cmd" --action colour %*
exit /b %ERRORLEVEL%
