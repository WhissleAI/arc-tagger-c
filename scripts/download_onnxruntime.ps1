# Windows (PowerShell): download the prebuilt ONNX Runtime release into third_party\onnxruntime
param([string]$Platform = "win-x64", [string]$Version = "1.20.1")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$name = "onnxruntime-$Platform-$Version"
$url = "https://github.com/microsoft/onnxruntime/releases/download/v$Version/$name.zip"
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ([System.Guid]::NewGuid())
New-Item -ItemType Directory -Path $tmp | Out-Null
Invoke-WebRequest -Uri $url -OutFile (Join-Path $tmp "ort.zip")
Expand-Archive -Path (Join-Path $tmp "ort.zip") -DestinationPath $tmp
$dest = Join-Path $root "third_party\onnxruntime"
if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
Move-Item (Join-Path $tmp $name) $dest
Remove-Item -Recurse -Force $tmp
Write-Host "ONNX Runtime $Version ($Platform) -> $dest"
