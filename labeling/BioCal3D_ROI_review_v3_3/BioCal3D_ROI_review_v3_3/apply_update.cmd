@echo off
setlocal EnableExtensions DisableDelayedExpansion
set "BIOCAL3D_TARGET=%~1"
if defined BIOCAL3D_TARGET goto selected
echo Close the annotator before applying the update.
set /p "BIOCAL3D_TARGET=Paste the EXISTING BioCal3D_Annotator app folder: "
:selected
set "BIOCAL3D_TARGET=%BIOCAL3D_TARGET:"=%"
if not exist "%BIOCAL3D_TARGET%\biocal3d_workflow\annotator\biocal3d_annotator.py" goto invalid
if not exist "%BIOCAL3D_TARGET%\.labeling-env\python.exe" goto python_fallback
"%BIOCAL3D_TARGET%\.labeling-env\python.exe" "%~dp0install_update.py" "%BIOCAL3D_TARGET%"
goto finished
:python_fallback
python "%~dp0install_update.py" "%BIOCAL3D_TARGET%"
goto finished
:invalid
echo Invalid app folder. Select the folder containing start_labelling.bat and biocal3d_workflow.
pause
exit /b 1
:finished
if errorlevel 1 echo Update failed. Review the message above.
pause
