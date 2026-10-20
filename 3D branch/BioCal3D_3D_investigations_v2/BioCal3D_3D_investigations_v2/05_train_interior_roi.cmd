@echo off
call "%~dp0_run.cmd" --action interior %*
exit /b %ERRORLEVEL%
