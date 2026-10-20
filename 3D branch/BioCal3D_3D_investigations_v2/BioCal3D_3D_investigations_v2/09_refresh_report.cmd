@echo off
call "%~dp0_run.cmd" --action report %*
exit /b %ERRORLEVEL%
