@echo off
call "%~dp0_run.cmd" --action loso %*
exit /b %ERRORLEVEL%
