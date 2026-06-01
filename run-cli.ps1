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
  [double]$KickWindowMs = 150.0,

  [switch]$Score,
  [string]$ScorePath = "",
  [string]$ScorePdfPath = "",
  [switch]$NoScorePdf,
  [double]$ScoreTempo = 0,
  [string]$ScoreKey = "",
  [string]$ScoreTitle = ""
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
if ($Score) {
  $ArgsList += @("--score")
  if ($ScorePath) {
    $ArgsList += @("--score-path", $ScorePath)
  }
  if ($ScorePdfPath) {
    $ArgsList += @("--score-pdf-path", $ScorePdfPath)
  }
  if ($NoScorePdf) {
    $ArgsList += @("--no-score-pdf")
  }
  if ($ScoreTempo -gt 0) {
    $ArgsList += @("--score-tempo", "$ScoreTempo")
  }
  if ($ScoreKey) {
    $ArgsList += @("--score-key", $ScoreKey)
  }
  if ($ScoreTitle) {
    $ArgsList += @("--score-title", $ScoreTitle)
  }
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
