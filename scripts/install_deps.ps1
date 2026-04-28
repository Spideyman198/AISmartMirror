# AISmartMirror - dependency install script (Windows)
# Uses dlib-bin (pre-built) instead of dlib to avoid CMake build.

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ProjectRoot

Write-Host "Installing AISmartMirror dependencies..." -ForegroundColor Cyan

# 1. dlib-bin first (pre-built wheel, no CMake)
Write-Host "`n1. Installing dlib-bin..." -ForegroundColor Yellow
pip install dlib-bin
if ($LASTEXITCODE -ne 0) { exit 1 }

# 2. face-recognition without pulling dlib
Write-Host "`n2. Installing face-recognition (no-deps)..." -ForegroundColor Yellow
pip install "face-recognition>=1.3.0" --no-deps
if ($LASTEXITCODE -ne 0) { exit 1 }

# 3. Rest of requirements
Write-Host "`n3. Installing core requirements..." -ForegroundColor Yellow
pip install -r requirements-base.txt
if ($LASTEXITCODE -ne 0) { exit 1 }

# 4. Optional: CNN requirements (use -All to install without prompting)
if ($args -contains "-All") {
    Write-Host "`n4. Installing CNN requirements..." -ForegroundColor Yellow
    pip install -r requirements-cnn.txt
} else {
    $installCnn = Read-Host "`nInstall CNN requirements (torch, torchvision)? [y/N]"
    if ($installCnn -eq "y" -or $installCnn -eq "Y") {
        Write-Host "`n4. Installing CNN requirements..." -ForegroundColor Yellow
        pip install -r requirements-cnn.txt
    }
}

Write-Host "`nDone." -ForegroundColor Green
