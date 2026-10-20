@echo off
call "%~dp0create_labeling_plan.cmd" %*
exit /b %ERRORLEVEL%
