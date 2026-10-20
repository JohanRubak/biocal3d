@echo off
call "%~dp0_run.cmd" --action preflight %*
exit /b %ERRORLEVEL%
