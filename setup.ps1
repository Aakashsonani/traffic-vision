$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

if (Get-Command uv -ErrorAction SilentlyContinue) {
    & uv sync --locked
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
} else {
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        if (Get-Command py -ErrorAction SilentlyContinue) {
            & py -3.12 -m venv .venv
        } elseif (Get-Command python -ErrorAction SilentlyContinue) {
            & python -m venv .venv
        } else {
            throw 'Install Python 3.12 from python.org, then run setup.ps1 again.'
        }
        if ($LASTEXITCODE -ne 0) { throw 'Could not create a Python environment.' }
    }
    & .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
}
& .\.venv\Scripts\python.exe scripts\download_models.py
if ($LASTEXITCODE -ne 0) { throw 'Model download failed. Check your connection and run setup again.' }
Write-Host 'Ready. Double-click start.bat to open Traffic Vision.'
