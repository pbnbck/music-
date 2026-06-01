$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
  throw "Run install.ps1 first."
}

Push-Location $Root
try {
  & $Python -m pip install -r requirements-dev.txt
  & $Python -m PyInstaller --noconfirm --name BassExtractorPro --onefile --windowed gui_launcher.py
  Write-Host "Build output: $Root\dist\BassExtractorPro.exe"
} finally {
  Pop-Location
}
