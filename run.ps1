<#
.SYNOPSIS
    Interview AI Studio - All-in-One Launcher for Windows (PowerShell)
.DESCRIPTION
    Automatically sets up virtual environment, installs dependencies via uv/pip,
    and runs batch video processing.
.EXAMPLE
    .\run.ps1
    .\run.ps1 --input D:\Videos --output D:\Output --min-silence 0.5
#>

[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ProcessArgs
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host " Interview AI Studio - Windows Batch Processor Launcher" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

# 1. Check for Python
$PythonCmd = $null
if (Get-Command "python" -ErrorAction SilentlyContinue) {
    $PythonCmd = "python"
} elseif (Get-Command "py" -ErrorAction SilentlyContinue) {
    $PythonCmd = "py"
} else {
    Write-Error "[ERROR] Python was not found in PATH. Please install Python 3.10+ from python.org."
    exit 1
}

# 2. Check for FFmpeg
if (-not (Get-Command "ffmpeg" -ErrorAction SilentlyContinue)) {
    Write-Warning "[WARN] FFmpeg was not found in PATH. Video rendering may fail without FFmpeg."
    Write-Warning "       Please install FFmpeg (e.g. winget install Gyan.FFmpeg or download from ffmpeg.org)."
}

# 3. Detect uv or fallback to standard venv/pip
$UseUv = $false
if (Get-Command "uv" -ErrorAction SilentlyContinue) {
    $UseUv = $true
}

$VenvDir = Join-Path $ScriptDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

# 4. Create virtual environment if missing
if (-not (Test-Path $VenvPython)) {
    Write-Host "[SETUP] Initializing virtual environment in .venv..." -ForegroundColor Yellow
    if ($UseUv) {
        Write-Host "[SETUP] Using uv for accelerated environment creation..." -ForegroundColor DarkGray
        uv venv "$VenvDir"
    } else {
        & $PythonCmd -m venv "$VenvDir"
    }
}

# 5. Install / verify dependencies
$ReqFile = Join-Path $ScriptDir "requirements.txt"
$FlagFile = Join-Path $VenvDir ".installed_requirements"

$NeedsInstall = $true
if (Test-Path $FlagFile) {
    $ReqHash = (Get-FileHash -Algorithm MD5 $ReqFile).Hash
    $SavedHash = Get-Content $FlagFile -Raw
    if ($ReqHash -eq $SavedHash) {
        $NeedsInstall = $false
    }
}

if ($NeedsInstall) {
    Write-Host "[SETUP] Installing required dependencies from requirements.txt..." -ForegroundColor Yellow
    if ($UseUv) {
        uv pip install --python "$VenvPython" -r "$ReqFile"
    } else {
        & $VenvPython -m pip install --upgrade pip
        & $VenvPython -m pip install -r "$ReqFile"
    }
    $ReqHash = (Get-FileHash -Algorithm MD5 $ReqFile).Hash
    Set-Content -Path $FlagFile -Value $ReqHash
    Write-Host "[SETUP] Dependencies installed successfully." -ForegroundColor Green
}

# 6. Resolve input/output arguments
$TargetArgs = @()
if ($ProcessArgs -and $ProcessArgs.Count -gt 0) {
    $TargetArgs = $ProcessArgs
} else {
    $InputDir = Join-Path $ScriptDir "inputs"
    $OutputDir = Join-Path $ScriptDir "outputs"
    if (-not (Test-Path $InputDir)) { New-Item -ItemType Directory -Path $InputDir -Force | Out-Null }
    if (-not (Test-Path $OutputDir)) { New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null }

    Write-Host "[INFO] No arguments provided. Defaulting to:" -ForegroundColor DarkCyan
    Write-Host "       Input  : $InputDir" -ForegroundColor DarkCyan
    Write-Host "       Output : $OutputDir" -ForegroundColor DarkCyan
    Write-Host "       (Place raw video files in ./inputs/ to process)" -ForegroundColor DarkGray

    $TargetArgs = @("--input", $InputDir, "--output", $OutputDir)
}

# 7. Execute processor
$ProcessorScript = Join-Path $ScriptDir "interview_processor.py"
Write-Host "`n[RUN] Starting Interview AI Studio processor..." -ForegroundColor Cyan
& $VenvPython $ProcessorScript @TargetArgs
$ExitCode = $LASTEXITCODE

if ($ExitCode -eq 0) {
    Write-Host "`n[DONE] Processing batch finished successfully." -ForegroundColor Green
} else {
    Write-Host "`n[FAIL] Processing failed with exit code $ExitCode." -ForegroundColor Red
}

exit $ExitCode
