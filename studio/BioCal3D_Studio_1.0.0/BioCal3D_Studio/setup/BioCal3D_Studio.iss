#define AppName "BioCal3D Studio"
#define AppVersion "1.0.0"
[Setup]
AppId={{1421762E-D096-45AA-99A5-90A7C356887D}
AppName={#AppName}
AppVersion={#AppVersion}
DefaultDirName={localappdata}\Programs\BioCal3D Studio
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\release
OutputBaseFilename=BioCal3D_Studio_Setup_1.0.0_x64
SetupIconFile=..\assets\biocal3d_studio.ico
UninstallDisplayIcon={app}\BioCal3D Studio.exe
Compression=lzma2
SolidCompression=yes
CloseApplications=yes
[Files]
Source: "..\dist\BioCal3D Studio\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
[Icons]
Name: "{autoprograms}\BioCal3D Studio"; Filename: "{app}\BioCal3D Studio.exe"
Name: "{autodesktop}\BioCal3D Studio"; Filename: "{app}\BioCal3D Studio.exe"
[Run]
Filename: "{app}\BioCal3D Studio.exe"; Description: "Open BioCal3D Studio"; Flags: nowait postinstall skipifsilent
