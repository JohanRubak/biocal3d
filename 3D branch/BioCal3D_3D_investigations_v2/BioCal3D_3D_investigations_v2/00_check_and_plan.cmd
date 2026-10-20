@echo off
call "%~dp0_run.cmd" --action audit --plan %*
exit /b %ERRORLEVEL%
