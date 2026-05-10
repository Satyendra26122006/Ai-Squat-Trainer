[Setup]
AppName=AI Squat Trainer
AppVersion=1.0
DefaultDirName={pf}\AI Squat Trainer
DefaultGroupName=AI Squat Trainer
DisableProgramGroupPage=yes
OutputBaseFilename=squat_trainer_installer
Compression=lzma
SolidCompression=yes
PrivilegesRequired=admin
OutputDir=.

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "dist\squat_trainer.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\AI Squat Trainer"; Filename: "{app}\squat_trainer.exe"
Name: "{userdesktop}\AI Squat Trainer"; Filename: "{app}\squat_trainer.exe"

[Run]
Filename: "{app}\squat_trainer.exe"; Description: "Launch AI Squat Trainer"; Flags: nowait postinstall skipifsilent
