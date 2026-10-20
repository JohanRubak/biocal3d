@echo off
call "%~dp0_run.cmd" --action audit %*
exit /b %ERRORLEVEL%
