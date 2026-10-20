@echo off
rem Compatibility name for the earlier G5 folder-preparation helper.
call "%~dp0..\create_labeling_plan.cmd" %*
exit /b %ERRORLEVEL%
