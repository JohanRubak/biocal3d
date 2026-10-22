@echo off
setlocal
cd /d "%~dp0"
echo BioCal3D Approved ROI exploration
echo Run this from Anaconda Prompt after: conda activate biocal3d
echo.
set "ROI_ROOT=C:\Users\au662213\repos\biocal3d\data\Approved ROIs"
set /p "ROI_INPUT=Approved ROIs folder [press Enter for %ROI_ROOT%]: "
if defined ROI_INPUT set "ROI_ROOT=%ROI_INPUT%"
python "%~dp0investigate_approved_rois.py" --data-root "%ROI_ROOT%" %*
set "ROI_EXIT=%ERRORLEVEL%"
echo.
if "%ROI_EXIT%"=="0" echo Analysis finished. Open report.html in the output folder printed above.
if "%ROI_EXIT%"=="2" echo Analysis finished with failed scans. Review tables\failures.csv before using results.
if "%ROI_EXIT%"=="1" echo Analysis stopped. Read the error above. For dependencies run: python -m pip install -r requirements.txt
pause
exit /b %ROI_EXIT%
