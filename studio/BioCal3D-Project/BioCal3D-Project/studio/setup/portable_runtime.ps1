# App-owned Python runtime. Compatible with Windows PowerShell 5.1.
function Get-VerifiedDownload {
    param([string]$Uri, [string]$Path, [string]$Sha256)
    if (Test-Path -LiteralPath $Path) {
        if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -eq $Sha256) { return }
        Remove-Item -LiteralPath $Path -Force
    }
    $Part = $Path + '.partial'
    for ($Attempt = 1; $Attempt -le 3; $Attempt++) {
        try {
            Invoke-WebRequest -UseBasicParsing -Uri $Uri -OutFile $Part -TimeoutSec 120
            if ((Get-FileHash -LiteralPath $Part -Algorithm SHA256).Hash -ne $Sha256) {
                throw "Downloaded file failed its SHA-256 check: $Uri"
            }
            Move-Item -LiteralPath $Part -Destination $Path -Force
            return
        } catch {
            Remove-Item -LiteralPath $Part -Force -ErrorAction SilentlyContinue
            if ($Attempt -eq 3) { throw }
            Start-Sleep -Seconds 2
        }
    }
}

function Initialize-PortableRuntime {
    param([string]$InstallRoot)
    $Runtime = Join-Path $InstallRoot 'runtime-3.11.9-x64'
    $Python = Join-Path $Runtime 'python.exe'
    $Marker = Join-Path $Runtime 'biocal3d-runtime-v1.txt'
    if ((Test-Path -LiteralPath $Python) -and (Test-Path -LiteralPath $Marker)) {
        & $Python -I -m pip --version | Out-Host
        if ($LASTEXITCODE -eq 0) { return $Runtime }
        throw "The private runtime is damaged. Rename $Runtime and run setup again."
    }
    if (Test-Path -LiteralPath $Runtime) {
        throw "An incomplete runtime exists at $Runtime. Rename this folder and run setup again."
    }
    $Downloads = Join-Path $InstallRoot 'downloads'
    New-Item -ItemType Directory -Force -Path $Downloads | Out-Null
    $Archive = Join-Path $Downloads 'python-3.11.9-embed-amd64.zip'
    $Pip = Join-Path $Downloads 'pip-25.0.1-py3-none-any.whl'
    Get-VerifiedDownload 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip' $Archive '009d6bf7e3b2ddca3d784fa09f90fe54336d5b60f0e0f305c37f400bf83cfd3b'
    Get-VerifiedDownload 'https://files.pythonhosted.org/packages/c9/bc/b7db44f5f39f9d0494071bddae6880eb645970366d0a200022a1a93d57f5/pip-25.0.1-py3-none-any.whl' $Pip 'c46efd13b6aa8279f33f2864459c8ce587ea6a1a59ee20de055868d8f7688f7f'
    $Stage = Join-Path $InstallRoot ('runtime-new-' + [guid]::NewGuid().ToString('N'))
    try {
        Expand-Archive -LiteralPath $Archive -DestinationPath $Stage
        $Packages = Join-Path $Stage 'Lib\site-packages'
        New-Item -ItemType Directory -Force -Path $Packages | Out-Null
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        [IO.Compression.ZipFile]::ExtractToDirectory($Pip, $Packages)
        $Paths = "python311.zip`r`n.`r`nLib\site-packages`r`nimport site`r`n"
        [IO.File]::WriteAllText((Join-Path $Stage 'python311._pth'), $Paths, [Text.Encoding]::ASCII)
        $StagePython = Join-Path $Stage 'python.exe'
        & $StagePython -I -m pip --version | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw 'The portable runtime could not start. Check Windows updates, antivirus messages and the setup log.'
        }
        [IO.File]::WriteAllText((Join-Path $Stage 'biocal3d-runtime-v1.txt'), 'Python 3.11.9 x64; pip 25.0.1')
        Move-Item -LiteralPath $Stage -Destination $Runtime
    } finally {
        if (Test-Path -LiteralPath $Stage) { Remove-Item -LiteralPath $Stage -Recurse -Force }
    }
    return $Runtime
}
