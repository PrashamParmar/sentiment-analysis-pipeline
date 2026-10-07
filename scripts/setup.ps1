# One-shot Windows setup. Keeps EVERY cache on the project drive (C: on this machine has ~2 GB free).
# Usage (from the project root):  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 [-Gpu]
param([switch]$Gpu)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
foreach ($d in ".cache\pip", ".cache\tmp", ".cache\hf") { New-Item -ItemType Directory -Force (Join-Path $root $d) | Out-Null }
$env:TMP = Join-Path $root ".cache\tmp"; $env:TEMP = $env:TMP
$env:PIP_CACHE_DIR = Join-Path $root ".cache\pip"
$env:HF_HOME = Join-Path $root ".cache\hf"

$venvPy = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { py -3.12 -m venv (Join-Path $root ".venv") }
& $venvPy -m pip install --upgrade pip
$index = if ($Gpu) { "https://download.pytorch.org/whl/cu128" } else { "https://download.pytorch.org/whl/cpu" }
& $venvPy -m pip install torch==2.14.1 --index-url $index
& $venvPy -m pip install -r (Join-Path $root "requirements.txt")
& $venvPy -m pip install -e $root --no-deps

# Persist cache locations for future shells (user scope, no admin needed).
[Environment]::SetEnvironmentVariable("HF_HOME", $env:HF_HOME, "User")
[Environment]::SetEnvironmentVariable("PIP_CACHE_DIR", $env:PIP_CACHE_DIR, "User")
Write-Host "Done. Activate with: .\.venv\Scripts\Activate.ps1"
