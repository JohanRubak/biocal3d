@echo off
call "%~dp0_run.cmd" --action colour --mature-only %*
exit /b %ERRORLEVEL%
