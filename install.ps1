$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PIP_CACHE_DIR = Join-Path $Root ".pip-cache"
New-Item -ItemType Directory -Force -Path $env:PIP_CACHE_DIR | Out-Null

function Invoke-Checked {
  param(
    [Parameter(Mandatory = $true)]
    [string]$File,
    [Parameter(Mandatory = $true)]
    [string[]]$Arguments
  )
  & $File @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed with exit code ${LASTEXITCODE}: $File $($Arguments -join ' ')"
  }
}

$PythonCandidates = @(
  "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
  "python",
  "py"
)

$Python = $null
foreach ($Candidate in $PythonCandidates) {
  try {
    & $Candidate --version *> $null
    if ($LASTEXITCODE -eq 0) {
      $Python = $Candidate
      break
    }
  } catch {
  }
}

if (-not $Python) {
  throw "Python 3.10+ was not found. Install Python first, then rerun install.ps1."
}

$CoreMissingPatterns = @(
  "MISSING  demucs",
  "MISSING  numpy",
  "MISSING  scipy",
  "MISSING  torch"
)

function Test-CoreDoctor {
  param(
    [Parameter(Mandatory = $true)]
    [string]$PythonPath
  )
  $Output = & $PythonPath -m bass_extractor.cli --doctor 2>&1
  $Exit = $LASTEXITCODE
  $Output | ForEach-Object { Write-Host $_ }
  foreach ($Pattern in $CoreMissingPatterns) {
    if (($Output -join "`n").Contains($Pattern)) {
      return $false
    }
  }
  return $true
}

Push-Location $Root
try {
  Invoke-Checked $Python @("-m", "venv", ".venv")
  Invoke-Checked ".\.venv\Scripts\python.exe" @("-m", "pip", "install", "--no-cache-dir", "--upgrade", "pip")
  Invoke-Checked ".\.venv\Scripts\python.exe" @("-m", "pip", "install", "--no-cache-dir", "-r", "requirements.txt")
  $CoreReady = Test-CoreDoctor ".\.venv\Scripts\python.exe"

  if (-not $CoreReady) {
    Write-Host ""
    Write-Host "Primary venv is not usable. Trying a system-site fallback that reuses an existing PyTorch install..."
    Invoke-Checked $Python @("-m", "venv", "--system-site-packages", ".venv-system")
    Invoke-Checked ".\.venv-system\Scripts\python.exe" @("-m", "pip", "install", "--no-cache-dir", "demucs==4.0.1", "torchaudio==0.13.1", "soundfile>=0.12")
    $CoreReady = Test-CoreDoctor ".\.venv-system\Scripts\python.exe"
  }

  if (-not $CoreReady) {
    throw "Core audio separation dependencies are still missing. See the diagnostics above."
  }

  Write-Host ""
  Write-Host "Install complete. Start the GUI with .\run-gui.ps1"
  Write-Host "If ffmpeg is missing, WAV input/output still works; install ffmpeg for MP3/M4A/AAC/OGG."
} finally {
  Pop-Location
}
