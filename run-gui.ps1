$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonCandidates = @(
  (Join-Path $Root ".venv-system\Scripts\python.exe"),
  (Join-Path $Root ".venv\Scripts\python.exe"),
  "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"
)
$Python = $PythonCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Python) { throw "Python was not found. Run install.ps1 first." }
Push-Location $Root
try {
  & $Python -m bass_extractor.gui
} finally {
  Pop-Location
}
