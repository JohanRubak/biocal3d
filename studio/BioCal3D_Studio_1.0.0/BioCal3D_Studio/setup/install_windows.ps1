param([Parameter(Mandatory=$true)][string]$Source)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
if (-not [Environment]::Is64BitOperatingSystem) { throw 'BioCal3D Studio requires a 64-bit operating system.' }
if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64' -or $env:PROCESSOR_ARCHITEW6432 -eq 'ARM64') { throw 'This installer targets Windows x64. Windows ARM has not been validated; use an x64 PC for this package.' }
$Source = (Resolve-Path -LiteralPath $Source).Path
$InstallRoot = Join-Path $env:LOCALAPPDATA 'BioCal3DStudio'
$LogRoot = Join-Path $InstallRoot 'logs'
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
$Log = Join-Path $LogRoot ('setup-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
Start-Transcript -Path $Log -Force | Out-Null
try {
    Write-Host 'BioCal3D Studio - per-user installation (no administrator account needed)'
    if (-not (Test-Path -LiteralPath (Join-Path $Source 'launch_studio.py'))) { throw 'Extract the complete ZIP before running setup.' }
    $Running = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^pythonw?\.exe$' -and $_.CommandLine -like '*BioCal3DStudio*launch_studio.py*' }
    if ($Running) { throw 'Close BioCal3D Studio before installing an update.' }
    $PythonRoot = Join-Path $InstallRoot 'python'
    $Python = Join-Path $PythonRoot 'python.exe'
    if (-not (Test-Path -LiteralPath $Python)) {
        $Installer = Join-Path $InstallRoot 'python-3.11.9-amd64.exe'
        Write-Host 'Downloading the private Python runtime...'
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -UseBasicParsing -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile $Installer
        $Signature = Get-AuthenticodeSignature -LiteralPath $Installer
        if ($Signature.Status -ne 'Valid' -or $Signature.SignerCertificate.Subject -notmatch 'Python Software Foundation') { throw 'The Python installer signature could not be verified. The installer was not run.' }
        $Arguments = '/quiet InstallAllUsers=0 PrependPath=0 Include_launcher=0 Include_test=0 Include_doc=0 Include_tcltk=0 Include_pip=1 TargetDir="' + $PythonRoot + '"'
        $Process = Start-Process -FilePath $Installer -ArgumentList $Arguments -Wait -PassThru
        if ($Process.ExitCode -notin @(0,3010)) { throw "Python installation failed (exit $($Process.ExitCode))." }
        if (-not (Test-Path -LiteralPath $Python)) { throw 'Python installation did not create the expected runtime.' }
        Remove-Item -LiteralPath $Installer -Force
    }
    Write-Host 'Installing the shared scientific and graphics dependencies. First setup may take several minutes...'
    & $Python -I -m pip install --disable-pip-version-check --only-binary=:all: -r (Join-Path $Source 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check your internet connection and the setup log.' }
    & $Python -I (Join-Path $Source 'setup\check_dependencies.py')
    if ($LASTEXITCODE -ne 0) { throw 'A dependency could not be imported.' }
    $Destination = Join-Path $InstallRoot 'application'
    $Stage = Join-Path $InstallRoot ('application-new-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Force -Path $Stage | Out-Null
    foreach ($Name in @('app','assets','launch_studio.py','requirements.txt','README.md')) {
        Copy-Item -LiteralPath (Join-Path $Source $Name) -Destination $Stage -Recurse -Force
    }
    $Backup = Join-Path $InstallRoot 'application-previous'
    if (Test-Path -LiteralPath $Backup) { Remove-Item -LiteralPath $Backup -Recurse -Force }
    if (Test-Path -LiteralPath $Destination) { Move-Item -LiteralPath $Destination -Destination $Backup }
    Move-Item -LiteralPath $Stage -Destination $Destination
    $Launch = Join-Path $InstallRoot 'Start_BioCal3D_Studio.cmd'
    $LaunchText = '@echo off' + "`r`n" + '"' + $Python + '" "' + (Join-Path $Destination 'launch_studio.py') + '" %*' + "`r`n" + 'if errorlevel 1 pause' + "`r`n"
    [IO.File]::WriteAllText($Launch,$LaunchText,[Text.Encoding]::Default)
    $Shell = New-Object -ComObject WScript.Shell
    foreach ($Folder in @($Shell.SpecialFolders.Item('Desktop'),$Shell.SpecialFolders.Item('Programs'))) {
        $Shortcut = $Shell.CreateShortcut((Join-Path $Folder 'BioCal3D Studio.lnk'))
        $Shortcut.TargetPath = Join-Path $PythonRoot 'pythonw.exe'
        $Shortcut.Arguments = '"' + (Join-Path $Destination 'launch_studio.py') + '"'
        $Shortcut.WorkingDirectory = $Destination
        $Shortcut.IconLocation = Join-Path $Destination 'assets\biocal3d_studio.ico'
        $Shortcut.Description = 'Inspect, orient and label the BioCal3D dataset'
        $Shortcut.Save()
    }
    Write-Host 'Installation complete. Open BioCal3D Studio using the desktop shortcut.'
    Write-Host "Log: $Log"
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host "Setup log: $Log"
    exit 1
} finally {
    Stop-Transcript | Out-Null
}
