@echo off
if /i "%CONDA_DEFAULT_ENV%"=="biocal3d" exit /b 0
where conda >nul 2>nul
if not errorlevel 1 goto found
for %%P in ("%USERPROFILE%\anaconda3\condabin\conda.bat" "%USERPROFILE%\miniconda3\condabin\conda.bat" "%LOCALAPPDATA%\anaconda3\condabin\conda.bat" "%LOCALAPPDATA%\miniconda3\condabin\conda.bat" "C:\ProgramData\anaconda3\condabin\conda.bat" "C:\ProgramData\miniconda3\condabin\conda.bat") do if exist "%%~P" goto usepath
 echo Cannot find conda. Open Anaconda Prompt, activate biocal3d, then run again.
exit /b 1
:found
call conda activate biocal3d
exit /b %ERRORLEVEL%
:usepath
REM Locate again through a subroutine so the chosen path is explicit.
for %%P in ("%USERPROFILE%\anaconda3\condabin\conda.bat" "%USERPROFILE%\miniconda3\condabin\conda.bat" "%LOCALAPPDATA%\anaconda3\condabin\conda.bat" "%LOCALAPPDATA%\miniconda3\condabin\conda.bat" "C:\ProgramData\anaconda3\condabin\conda.bat" "C:\ProgramData\miniconda3\condabin\conda.bat") do if exist "%%~P" call :activate "%%~P"
if /i "%CONDA_DEFAULT_ENV%"=="biocal3d" exit /b 0
exit /b 1
:activate
if /i "%CONDA_DEFAULT_ENV%"=="biocal3d" exit /b 0
call "%~1" activate biocal3d
exit /b %ERRORLEVEL%
