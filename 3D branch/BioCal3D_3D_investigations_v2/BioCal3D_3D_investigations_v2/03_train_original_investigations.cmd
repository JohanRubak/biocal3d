@echo off
call "%~dp0_run.cmd" --action baseline %*
exit /b %ERRORLEVEL%
