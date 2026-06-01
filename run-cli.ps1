param(
  [Parameter(Mandatory = $true)]
  [string]$InputSong,

  [string]$OutputPath = "",
  [ValidateSet("studio", "balanced", "fast")]
  [string]$Profile = "studio",
  [ValidateSet("auto", "cpu", "cuda")]
  [string]$Device = "auto",
  [ValidateSet("wav", "flac", "mp3")]
  [string]$Format = "wav",

  [switch]$KickClean,
  [double]$KickStrength = 0.65,
  [double]$KickMinFrequency = 35.0,
  [double]$KickMaxFrequency = 135.0,
  [double]$KickWindowMs = 150.0
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonCandidates = @(
  (Join-Path $Root ".venv-system\Scripts\python.exe"),
  (Join-Path $Root ".venv\Scripts\python.exe"),
  "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"
)
$Python = $PythonCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Python) { throw "Python was not found. Run install.ps1 first." }

$ArgsList = @("-m", "bass_extractor.cli", $InputSong, "--profile", $Profile, "--device", $Device, "--format", $Format)
if ($OutputPath) {
  $ArgsList += @("-o", $OutputPath)
}
if ($KickClean) {
  $ArgsList += @(
    "--kick-clean",
    "--kick-strength", "$KickStrength",
    "--kick-min-frequency", "$KickMinFrequency",
    "--kick-max-frequency", "$KickMaxFrequency",
    "--kick-window-ms", "$KickWindowMs"
  )
}

Push-Location $Root
$ExitCode = 0
try {
  & $Python @ArgsList
  $ExitCode = $LASTEXITCODE
} finally {
  Pop-Location
}
exit $ExitCode
