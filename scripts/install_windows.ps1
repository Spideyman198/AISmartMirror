# Windows install script - uses pre-built dlib to avoid CMake build
# Run from project root: .\scripts\install_windows.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

Write-Host "Installing dlib-bin (pre-built wheel)..." -ForegroundColor Cyan
pip install dlib-bin

Write-Host "Installing face_recognition (skipping dlib build)..." -ForegroundColor Cyan
pip install face_recognition --no-deps

Write-Host "Installing remaining dependencies..." -ForegroundColor Cyan
pip install face-recognition-models "Click>=6.0"
pip install -r requirements.txt

Write-Host "Done. Run: python -m app.main" -ForegroundColor Green
